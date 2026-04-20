/// <reference lib="webworker" />

declare function importScripts(...urls: string[]): void;
declare function loadPyodide(options: { indexURL: string }): Promise<any>;

const PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/';

const MODEL_FILES = [
  'math_utils.py',
  'generative_model.py',
  'dirichlet.py',
  'free_energy.py',
  'policy.py',
  'belief.py',
  'agents.py',
  'simulation.py',
  'maze_model.py',
  'maze_efe.py',
  'maze_simulation.py',
  'scenario_runner.py',
];

const SCENARIO_FILES = [
  'scenarios/__init__.py',
  'scenarios/tmaze.py',
  'scenarios/grid_maze.py',
  'scenarios/drone_search.py',
];

type BenchmarkModelParams = {
  rewardProb: number;
  contextProb: number;
  rewardMag: number;
};

type BenchmarkBatchRequest = {
  type: 'benchmark-batch';
  requestId: number;
  baseUrl: string;
  scenarioId: string;
  nSteps: number;
  episodesPerAgent: number;
  agentTypes: string[];
  modelParams?: BenchmarkModelParams;
  forceExtrinsicOnly?: boolean;
};

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;

let pyodideReady: Promise<any> | null = null;
let queuedWork: Promise<void> = Promise.resolve();

function escapeJsonForPython(value: unknown): string {
  return JSON.stringify(value)
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"');
}

async function loadModelIntoWorker(pyodide: any, baseUrl: string): Promise<void> {
  const fs = pyodide.FS;
  try { fs.mkdir('/model'); } catch { /* already exists */ }
  try { fs.mkdir('/model/scenarios'); } catch { /* already exists */ }

  for (const filename of MODEL_FILES) {
    const response = await fetch(new URL(`model/${filename}`, baseUrl).toString());
    if (!response.ok) throw new Error(`Could not load model/${filename}`);
    fs.writeFile(`/model/${filename}`, await response.text());
  }

  for (const filename of SCENARIO_FILES) {
    const response = await fetch(new URL(`model/${filename}`, baseUrl).toString());
    if (!response.ok) throw new Error(`Could not load model/${filename}`);
    fs.writeFile(`/model/${filename}`, await response.text());
  }

  await pyodide.runPythonAsync(`
import sys
if '/model' not in sys.path:
    sys.path.insert(0, '/model')
  `);

  await pyodide.loadPackage('scipy');
  await pyodide.runPythonAsync('from scenario_runner import runner');
}

async function ensurePyodide(baseUrl: string): Promise<any> {
  if (!pyodideReady) {
    pyodideReady = (async () => {
      importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
      const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
      await pyodide.loadPackage('numpy');
      await loadModelIntoWorker(pyodide, baseUrl);
      return pyodide;
    })();
    pyodideReady.catch(() => {
      pyodideReady = null;
    });
  }
  return pyodideReady;
}

async function handleBenchmarkBatch(request: BenchmarkBatchRequest): Promise<void> {
  const pyodide = await ensurePyodide(request.baseUrl);
  const scenarioLiteral = JSON.stringify(request.scenarioId);
  const agentTypesJson = escapeJsonForPython(request.agentTypes);
  const modelUpdate = request.modelParams
    ? `runner.set_model_params(${request.modelParams.rewardProb}, ${request.modelParams.contextProb}, ${request.modelParams.rewardMag})`
    : '';

  const result = pyodide.runPython(`
import json
runner.switch(${scenarioLiteral})
${modelUpdate}
json.dumps(runner.benchmark_batch(${request.nSteps}, ${request.episodesPerAgent}, json.loads("${agentTypesJson}"), force_extrinsic_only=${request.forceExtrinsicOnly ? 'True' : 'False'}))
  `);

  ctx.postMessage({
    type: 'benchmark-result',
    requestId: request.requestId,
    batch: JSON.parse(result),
  });
}

ctx.onmessage = (event: MessageEvent<BenchmarkBatchRequest>) => {
  const request = event.data;
  if (request.type !== 'benchmark-batch') return;

  queuedWork = queuedWork
    .then(() => handleBenchmarkBatch(request))
    .catch((error: unknown) => {
      ctx.postMessage({
        type: 'benchmark-error',
        requestId: request.requestId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
};