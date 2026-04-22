declare global {
  interface Window {
    loadPyodide: (options: { indexURL: string }) => Promise<any>;
  }
}

type ProgressCb = (msg: string, pct: number) => void;

export type BenchmarkModelParams = {
  rewardProb: number;
  contextProb: number;
  rewardMag: number;
};

type PendingBenchmarkRequest = {
  resolve: (value: any) => void;
  reject: (reason?: unknown) => void;
};

// All Python model files needed for multi-scenario runner
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
  'scenarios/tmaze_stable.py',
  'scenarios/tmaze_learning.py',
  'scenarios/grid_maze.py',
  'scenarios/drone_search.py',
  'scenarios/drone_search_v2.py',
];

let benchmarkWorker: Worker | null = null;
let nextBenchmarkRequestId = 0;
const pendingBenchmarkRequests = new Map<number, PendingBenchmarkRequest>();

function rejectPendingBenchmarkRequests(error: unknown): void {
  for (const pending of pendingBenchmarkRequests.values()) {
    pending.reject(error);
  }
  pendingBenchmarkRequests.clear();
}

function ensureBenchmarkWorker(): Worker {
  if (benchmarkWorker) return benchmarkWorker;

  benchmarkWorker = new Worker(new URL('./benchmarkWorker.ts', import.meta.url));
  benchmarkWorker.onmessage = (event: MessageEvent<any>) => {
    const message = event.data;
    if (message?.type !== 'benchmark-result' && message?.type !== 'benchmark-error') return;

    const pending = pendingBenchmarkRequests.get(message.requestId);
    if (!pending) return;
    pendingBenchmarkRequests.delete(message.requestId);

    if (message.type === 'benchmark-result') {
      pending.resolve(message.batch);
      return;
    }

    pending.reject(new Error(message.error ?? 'Benchmark worker failed'));
  };
  benchmarkWorker.onerror = (event: ErrorEvent) => {
    rejectPendingBenchmarkRequests(new Error(event.message || 'Benchmark worker crashed'));
    benchmarkWorker?.terminate();
    benchmarkWorker = null;
  };

  return benchmarkWorker;
}

export function cancelBenchmarkWorkerJobs(): void {
  if (!benchmarkWorker) return;
  benchmarkWorker.terminate();
  benchmarkWorker = null;
  rejectPendingBenchmarkRequests(new Error('Benchmark worker canceled'));
}

export async function initPyodide(onProgress: ProgressCb): Promise<any> {
  onProgress('Downloading Pyodide runtime...', 10);

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Pyodide'));
    document.head.appendChild(script);
  });

  onProgress('Initialising Python...', 30);
  const pyodide = await window.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/',
  });

  onProgress('Loading NumPy...', 45);
  await pyodide.loadPackage('numpy');

  return pyodide;
}

export async function loadModel(pyodide: any): Promise<void> {
  const fs = pyodide.FS;
  try { fs.mkdir('/model'); } catch { /* already exists */ }
  try { fs.mkdir('/model/scenarios'); } catch { /* already exists */ }

  // Load core model files
  for (const filename of MODEL_FILES) {
    const url = `./model/${filename}`;
    const response = await fetch(url);
    if (!response.ok) { console.warn(`Could not load ${url}, skipping`); continue; }
    fs.writeFile(`/model/${filename}`, await response.text());
  }

  // Load scenario files
  for (const filename of SCENARIO_FILES) {
    const url = `./model/${filename}`;
    const response = await fetch(url);
    if (!response.ok) { console.warn(`Could not load ${url}, skipping`); continue; }
    fs.writeFile(`/model/${filename}`, await response.text());
  }

  await pyodide.runPythonAsync(`
import sys
if '/model' not in sys.path:
    sys.path.insert(0, '/model')
  `);

  await pyodide.loadPackage('scipy');

  // Import the scenario runner (imports everything transitively)
  await pyodide.runPythonAsync('from scenario_runner import runner');
}

// ─── Scenario runner bridge ───

export interface ScenarioInfo {
  id: string;
  name: string;
}

export function listScenarios(pyodide: any): ScenarioInfo[] {
  const result = pyodide.runPython('import json; json.dumps(runner.list_scenarios())');
  return JSON.parse(result);
}

export function switchScenario(pyodide: any, scenarioId: string): any {
  const result = pyodide.runPython(`import json; json.dumps(runner.switch("${scenarioId}"))`);
  return JSON.parse(result);
}

export function callStep(pyodide: any): any {
  const result = pyodide.runPython('import json; json.dumps(runner.step())');
  return JSON.parse(result);
}

export function callReset(pyodide: any, agentType: string): void {
  pyodide.runPython(`runner.reset("${agentType}")`);
}

export function callHardReset(pyodide: any, agentType: string): void {
  pyodide.runPython(`runner.hard_reset("${agentType}")`);
}

export function callRunExperiment(pyodide: any, nTrials: number): any[] {
  const result = pyodide.runPython(
    `import json; json.dumps(runner.run_experiment(${nTrials}))`
  );
  return JSON.parse(result);
}

/** Run N full episodes for the current scenario's agent, silently (no step animation).
 *  Returns per-episode summaries plus the post-training world_alpha if available. */
export function callTrainEpisodes(pyodide: any, agentType: string, nEpisodes: number, nSteps: number): any {
  const result = pyodide.runPython(`
import json
summaries = []
scen = runner.scenario
for _ in range(${nEpisodes}):
    scen.reset("${agentType}")
    summaries.append(scen.run_episode_summary(${nSteps}))
world_alpha = [float(v) for v in getattr(scen, 'world_alpha', [])]
json.dumps({'summaries': summaries, 'world_alpha': world_alpha})
`);
  return JSON.parse(result);
}

/** Run exactly one training episode. Safe to call repeatedly from the UI loop —
 *  each call yields back control so a progress bar can paint. */
export async function callTrainOneEpisode(pyodide: any, agentType: string, nSteps: number): Promise<any> {
  const result = await pyodide.runPythonAsync(
    `import json; json.dumps(runner.train_one_episode("${agentType}", ${nSteps}))`,
  );
  return JSON.parse(result);
}

export function callBenchmarkBatch(pyodide: any, nSteps: number, episodesPerAgent: number, agentTypes: string[]): any {
  const agentTypesJson = JSON.stringify(agentTypes)
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"');
  const result = pyodide.runPython(
    `import json; json.dumps(runner.benchmark_batch(${nSteps}, ${episodesPerAgent}, json.loads(\"${agentTypesJson}\")))`
  );
  return JSON.parse(result);
}

export async function callBenchmarkBatchAsync(request: {
  scenarioId: string;
  nSteps: number;
  episodesPerAgent: number;
  agentTypes: string[];
  modelParams?: BenchmarkModelParams;
  forceExtrinsicOnly?: boolean;
}): Promise<any> {
  const worker = ensureBenchmarkWorker();
  const requestId = ++nextBenchmarkRequestId;
  const baseUrl = new URL('.', window.location.href).toString();

  return new Promise((resolve, reject) => {
    pendingBenchmarkRequests.set(requestId, { resolve, reject });
    worker.postMessage({
      type: 'benchmark-batch',
      requestId,
      baseUrl,
      ...request,
    });
  });
}

export function setParams(pyodide: any, beta: number, wExt: number, wSal: number, wNov: number): void {
  pyodide.runPython(`runner.set_params(${beta}, ${wExt}, ${wSal}, ${wNov})`);
}

export function setModelParams(pyodide: any, rewardProb: number, contextProb: number, rewardMag: number): void {
  pyodide.runPython(`runner.set_model_params(${rewardProb}, ${contextProb}, ${rewardMag})`);
}
