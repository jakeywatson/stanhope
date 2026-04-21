import { initPyodide, loadModel, listScenarios, switchScenario, callStep, callReset, callHardReset, callRunExperiment, callTrainEpisodes, callBenchmarkBatchAsync, cancelBenchmarkWorkerJobs, setParams, setModelParams } from './bridge';
import type { BenchmarkModelParams } from './bridge';
import { ExperimentDashboard } from './ExperimentDashboard';
import type { SceneController, SceneObjects } from './scenarios/types';
import { TMazeScene } from './scenarios/TMazeScene';
import { GridMazeScene } from './scenarios/GridMazeScene';
import { DroneScene } from './scenarios/DroneScene';
import { DroneSceneV2 } from './scenarios/DroneSceneV2';

const progress = document.getElementById('progress-fill') as HTMLDivElement;
const status = document.getElementById('loading-status') as HTMLDivElement;
const loading = document.getElementById('loading') as HTMLDivElement;

// ─── Scenario registry ───
const SCENE_MAP: Record<string, () => SceneController> = {
  tmaze: () => new TMazeScene(),
  grid_maze: () => new GridMazeScene(),
  drone_search: () => new DroneScene(),
  drone_search_v2: () => new DroneSceneV2(),
};

// ─── Agent type options per scenario ───
const AGENTS: Record<string, { value: string; label: string }[]> = {
  tmaze: [
    { value: 'active_learning', label: 'Active Learning' },
    { value: 'active_inference', label: 'Active Inference' },
    { value: 'combined', label: 'Combined' },
    { value: 'greedy', label: 'Greedy' },
    { value: 'random', label: 'Random' },
  ],
  grid_maze: [
    { value: 'combined', label: 'Combined' },
    { value: 'active_learning', label: 'Active Learning' },
    { value: 'active_inference', label: 'Active Inference' },
    { value: 'greedy', label: 'Greedy' },
    { value: 'random', label: 'Random' },
  ],
  drone_search: [
    { value: 'combined', label: 'Combined' },
    { value: 'active_learning', label: 'Active Learning' },
    { value: 'active_inference', label: 'Active Inference' },
    { value: 'greedy', label: 'Greedy' },
    { value: 'random', label: 'Random' },
  ],
  drone_search_v2: [
    { value: 'combined', label: 'Combined' },
    { value: 'active_learning', label: 'Active Learning' },
    { value: 'active_inference', label: 'Active Inference' },
    { value: 'greedy', label: 'Greedy' },
    { value: 'random', label: 'Random' },
  ],
};

const DEFAULT_STEPS: Record<string, number> = {
  tmaze: 32,
  grid_maze: 20,
  drone_search: 200,
  drone_search_v2: 400,
};

const DEFAULT_BENCHMARK_SETTINGS: Record<string, { episodesPerAgent: number; batchSize: number }> = {
  tmaze: { episodesPerAgent: 120, batchSize: 20 },
  grid_maze: { episodesPerAgent: 40, batchSize: 10 },
  drone_search: { episodesPerAgent: 40, batchSize: 10 },
  drone_search_v2: { episodesPerAgent: 30, batchSize: 5 },
};

type ViewMode = 'interactive' | 'experimenter';

let pyodide: any;
let currentScenarioId = 'tmaze';
let currentAgent = 'combined';
let currentViewMode: ViewMode = 'interactive';
let controller: SceneController | null = null;
let sceneObjects: SceneObjects | null = null;
let experimentDashboard: ExperimentDashboard | null = null;
let animFrameId = 0;
let isRunning = false;
let cancelToken = 0;

// ─── Agent descriptions (paper terminology) ───
const AGENT_DESC: Record<string, { formula: string; desc: string }> = {
  combined: {
    formula: 'G = extrinsic + salience + novelty',
    desc: 'Full EFE agent — reward-seeking with strong cue-driven uncertainty reduction and a lighter novelty bonus.',
  },
  active_learning: {
    formula: 'G = extrinsic + novelty',
    desc: 'Balances payoff with parameter learning — explores uncertain arms while still caring about reward.',
  },
  active_inference: {
    formula: 'G = extrinsic + salience',
    desc: 'No parameter learning. Pragmatic — seeks reward and resolves ambiguity, but doesn\'t learn concentrations.',
  },
  greedy: {
    formula: 'G = extrinsic',
    desc: 'Pure exploiter — only chases expected reward. No curiosity or learning.',
  },
  random: {
    formula: 'G = uniform',
    desc: 'Baseline — picks actions at random. No planning at all.',
  },
};

// Preset slider values per agent type: [beta, w_ext, w_sal, w_nov]
const AGENT_PARAMS: Record<string, [number, number, number, number]> = {
  combined:         [0.75, 1.0, 2.0, 0.75],
  active_learning:  [0.75, 1.0, 0.0, 1.5],
  active_inference: [0.75, 1.0, 2.0, 0.0],
  greedy:           [0.125, 1.0, 0.0, 0.0],
  random:           [8.0, 1.0, 0.0, 0.0],
};

function syncSlidersToAgent(agent: string) {
  const params = AGENT_PARAMS[agent] ?? [1.0, 1.0, 1.0, 1.0];
  const [beta, ext, sal, nov] = params;
  (document.getElementById('sl-beta') as HTMLInputElement).value = String(beta);
  (document.getElementById('sl-ext') as HTMLInputElement).value = String(ext);
  (document.getElementById('sl-sal') as HTMLInputElement).value = String(sal);
  (document.getElementById('sl-nov') as HTMLInputElement).value = String(nov);
  document.getElementById('sv-beta')!.textContent = beta.toFixed(1);
  document.getElementById('sv-ext')!.textContent = ext.toFixed(2);
  document.getElementById('sv-sal')!.textContent = sal.toFixed(2);
  document.getElementById('sv-nov')!.textContent = nov.toFixed(2);
}

function pushSliderParams() {
  const beta = parseFloat((document.getElementById('sl-beta') as HTMLInputElement).value);
  const ext = parseFloat((document.getElementById('sl-ext') as HTMLInputElement).value);
  const sal = parseFloat((document.getElementById('sl-sal') as HTMLInputElement).value);
  const nov = parseFloat((document.getElementById('sl-nov') as HTMLInputElement).value);
  document.getElementById('sv-beta')!.textContent = beta.toFixed(1);
  document.getElementById('sv-ext')!.textContent = ext.toFixed(2);
  document.getElementById('sv-sal')!.textContent = sal.toFixed(2);
  document.getElementById('sv-nov')!.textContent = nov.toFixed(2);
  if (pyodide) setParams(pyodide, beta, ext, sal, nov);
}

function pushModelParams() {
  const rprob = parseFloat((document.getElementById('sl-rprob') as HTMLInputElement).value);
  const ctx = parseFloat((document.getElementById('sl-ctx') as HTMLInputElement).value);
  const rmag = parseFloat((document.getElementById('sl-rmag') as HTMLInputElement).value);
  document.getElementById('sv-rprob')!.textContent = rprob.toFixed(2);
  document.getElementById('sv-ctx')!.textContent = ctx.toFixed(2);
  document.getElementById('sv-rmag')!.textContent = rmag.toFixed(1);
  if (pyodide) setModelParams(pyodide, rprob, ctx, rmag);
}

function syncModelSliders() {
  (document.getElementById('sl-rprob') as HTMLInputElement).value = '0.90';
  (document.getElementById('sl-ctx') as HTMLInputElement).value = '0.50';
  (document.getElementById('sl-rmag') as HTMLInputElement).value = '4.0';
  document.getElementById('sv-rprob')!.textContent = '0.90';
  document.getElementById('sv-ctx')!.textContent = '0.50';
  document.getElementById('sv-rmag')!.textContent = '4.0';
}

function getBenchmarkModelParams(): BenchmarkModelParams | undefined {
  if (currentScenarioId !== 'tmaze') return undefined;

  return {
    rewardProb: parseFloat((document.getElementById('sl-rprob') as HTMLInputElement).value),
    contextProb: parseFloat((document.getElementById('sl-ctx') as HTMLInputElement).value),
    rewardMag: parseFloat((document.getElementById('sl-rmag') as HTMLInputElement).value),
  };
}

function showModelSliders(visible: boolean) {
  const el = document.getElementById('model-sliders');
  if (el) el.classList.toggle('visible', visible);
}

function updateAgentDesc() {
  const el = document.getElementById('agent-desc');
  if (!el) return;
  if (currentViewMode === 'experimenter') {
    el.innerHTML = '<span class="ad-name">experimenter</span> <span class="ad-efe">benchmark mode</span> <span style="margin-left:0.8rem">Runs all agents from fresh resets, updates cumulative charts live, and compares accuracy, reward, and efficiency for the selected scenario.</span>';
    return;
  }
  const info = AGENT_DESC[currentAgent];
  if (!info) { el.innerHTML = ''; return; }
  el.innerHTML = `<span class="ad-name">${currentAgent.replace('_', ' ')}</span> <span class="ad-efe">${info.formula}</span> <span style="margin-left:0.8rem">${info.desc}</span>`;
}

function setAgentControlsDisabled(disabled: boolean) {
  (document.getElementById('agent-select') as HTMLSelectElement).disabled = disabled;
  for (const id of ['sl-beta', 'sl-ext', 'sl-sal', 'sl-nov']) {
    (document.getElementById(id) as HTMLInputElement).disabled = disabled;
  }
}

function updateTopbarForMode() {
  const btnStep = document.getElementById('btn-step') as HTMLButtonElement;
  const btnRun = document.getElementById('btn-run') as HTMLButtonElement;
  const btnExperiment = document.getElementById('btn-experiment') as HTMLButtonElement;
  const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  const trialLabel = document.getElementById('trial-label');

  if (currentViewMode === 'experimenter') {
    btnStep.textContent = 'Run Batch';
    btnRun.textContent = 'Run Benchmark';
    btnExperiment.textContent = 'Clear Results';
    btnReset.textContent = 'Reset Scenario';
    if (trialLabel) trialLabel.textContent = 'Episodes';
    setAgentControlsDisabled(true);
  } else {
    btnStep.textContent = 'Step';
    btnRun.textContent = 'Run 10';
    btnExperiment.textContent = `Run ${DEFAULT_STEPS[currentScenarioId] ?? 32} Steps`;
    btnReset.textContent = 'Reset';
    if (trialLabel) trialLabel.textContent = 'Step';
    setAgentControlsDisabled(false);
  }
}

function disposeCurrentView() {
  if (controller) {
    controller.dispose();
    controller = null;
  }
  if (sceneObjects) {
    sceneObjects.renderer.dispose();
    sceneObjects.renderer.domElement.remove();
    sceneObjects = null;
  }
  if (experimentDashboard) {
    experimentDashboard.dispose();
    experimentDashboard = null;
  }
  cancelAnimationFrame(animFrameId);
  const viewport = document.getElementById('viewport') as HTMLDivElement | null;
  if (viewport) {
    // Preserve the ribbon across scene switches; clear everything else.
    const ribbon = document.getElementById('viewport-ribbon');
    viewport.innerHTML = '';
    if (ribbon) viewport.appendChild(ribbon);
  }
  setRibbon(null, null);
}

async function boot() {
  status.textContent = 'Loading Python runtime (Pyodide)...';
  progress.style.width = '15%';
  pyodide = await initPyodide((msg: string, pct: number) => {
    status.textContent = msg;
    progress.style.width = `${pct}%`;
  });

  status.textContent = 'Loading model...';
  progress.style.width = '60%';
  await loadModel(pyodide);

  // Populate scenario selector
  status.textContent = 'Building scene...';
  progress.style.width = '80%';
  const scenarios = listScenarios(pyodide);
  const scenarioSelect = document.getElementById('scenario-select') as HTMLSelectElement;
  // Hide grid_maze and the v1 drone from the interview demo — their stories
  // duplicate T-maze / are superseded by drone_search_v2. Expose with ?scenarios=all.
  const showAll = new URLSearchParams(window.location.search).get('scenarios') === 'all';
  const HIDDEN_BY_DEFAULT = new Set(['grid_maze', 'drone_search']);
  for (const s of scenarios) {
    if (!showAll && HIDDEN_BY_DEFAULT.has(s.id)) continue;
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.name;
    scenarioSelect.appendChild(opt);
  }

  // Init first scenario
  activateScenario(currentScenarioId, true);

  status.textContent = 'Ready.';
  progress.style.width = '100%';
  setTimeout(() => loading.classList.add('hidden'), 400);

  // ─── Event listeners ───
  scenarioSelect.addEventListener('change', () => {
    cancelToken++;          // cancel any in-flight run/step
    cancelBenchmarkWorkerJobs();
    isRunning = false;
    setButtons(true);
    activateScenario(scenarioSelect.value, true);
  });

  document.getElementById('mode-select')!.addEventListener('change', (e) => {
    cancelToken++;
    cancelBenchmarkWorkerJobs();
    isRunning = false;
    setButtons(true);
    currentViewMode = (e.target as HTMLSelectElement).value as ViewMode;
    activateScenario(currentScenarioId, false);
  });

  document.getElementById('agent-select')!.addEventListener('change', (e) => {
    currentAgent = (e.target as HTMLSelectElement).value;
    callHardReset(pyodide, currentAgent);
    syncSlidersToAgent(currentAgent);
    if (controller) { controller.reset(); controller.resetPanel(); }
    setRibbon(null, null);
    updateAgentDesc();
  });

  // Slider listeners — push params to Python on every change
  for (const id of ['sl-beta', 'sl-ext', 'sl-sal', 'sl-nov']) {
    document.getElementById(id)!.addEventListener('input', pushSliderParams);
  }
  for (const id of ['sl-rprob', 'sl-ctx', 'sl-rmag']) {
    document.getElementById(id)!.addEventListener('input', pushModelParams);
  }

  document.getElementById('btn-step')!.addEventListener('click', () => {
    if (isRunning) return;
    if (currentViewMode === 'experimenter') {
      doBenchmarkBatch();
    } else {
      doStep();
    }
  });

  document.getElementById('btn-run')!.addEventListener('click', () => {
    if (isRunning) return;
    if (currentViewMode === 'experimenter') {
      doBenchmark();
    } else {
      doRun();
    }
  });

  document.getElementById('btn-experiment')!.addEventListener('click', () => {
    if (isRunning) return;
    if (currentViewMode === 'experimenter') {
      clearBenchmarkResults();
    } else {
      doExperiment();
    }
  });

  document.getElementById('btn-reset')!.addEventListener('click', () => {
    if (isRunning) return;
    if (currentViewMode === 'experimenter') {
      callHardReset(pyodide, currentAgent);
      clearBenchmarkResults();
    } else {
      callReset(pyodide, currentAgent);
      if (controller) { controller.reset(); controller.resetPanel(); }
      setRibbon(null, null);
      updateTrialCounter(0);
    }
  });

  window.addEventListener('resize', onResize);
}

function activateScenario(scenarioId: string, scenarioChanged = false) {
  disposeCurrentView();

  currentScenarioId = scenarioId;
  if (scenarioChanged || !(AGENTS[scenarioId] ?? []).some(a => a.value === currentAgent)) {
    currentAgent = AGENTS[scenarioId]?.[0]?.value ?? 'combined';
  }
  if (scenarioChanged) {
    syncSlidersToAgent(currentAgent);
  }
  showModelSliders(scenarioId === 'tmaze');
  if (scenarioId === 'tmaze' && scenarioChanged) syncModelSliders();

  // Switch Python scenario
  const cfg = switchScenario(pyodide, scenarioId);
  callHardReset(pyodide, currentAgent);

  // Update agent dropdown
  const agentSelect = document.getElementById('agent-select') as HTMLSelectElement;
  agentSelect.innerHTML = '';
  for (const a of (AGENTS[scenarioId] ?? [])) {
    const opt = document.createElement('option');
    opt.value = a.value;
    opt.textContent = a.label;
    agentSelect.appendChild(opt);
  }
  agentSelect.value = currentAgent;

  updateTopbarForMode();
  updateAgentDesc();

  const viewport = document.getElementById('viewport') as HTMLDivElement;
  const scenarioPanel = document.getElementById('scenario-panel') as HTMLDivElement;

  if (currentViewMode === 'experimenter') {
    experimentDashboard = new ExperimentDashboard(scenarioId, cfg.scenario_name ?? scenarioId, AGENTS[scenarioId] ?? []);
    experimentDashboard.init(viewport);
    scenarioPanel.innerHTML = experimentDashboard.buildPanel({
      stepCap: DEFAULT_STEPS[scenarioId] ?? 32,
      ...(DEFAULT_BENCHMARK_SETTINGS[scenarioId] ?? { episodesPerAgent: 60, batchSize: 5 }),
    });
    experimentDashboard.attachPanel();
    experimentDashboard.setStatus('Ready to benchmark all agents.');
    updateTrialCounter(0);
    return;
  }

  // Create scene controller
  const factory = SCENE_MAP[scenarioId];
  if (!factory) { console.error(`No scene for ${scenarioId}`); return; }
  controller = factory();

  sceneObjects = controller.init(viewport);

  // Inject panel HTML
  scenarioPanel.innerHTML = controller.buildPanel();

  if (scenarioId === 'drone_search_v2') {
    wireDroneV2PanelControls();
  }

  updateTrialCounter(0);

  // Start render loop
  const { scene, camera, renderer, controls, onFrame } = sceneObjects;
  function animate() {
    animFrameId = requestAnimationFrame(animate);
    controls.update();
    onFrame();
    renderer.render(scene, camera);
  }
  animate();
}

async function doStep() {
  const token = cancelToken;
  isRunning = true;
  setButtons(false);
  try {
    const result = callStep(pyodide);
    if (token !== cancelToken) return;
    await controller!.animateStep(result);
    if (token !== cancelToken) return;
    controller!.updatePanel(result);
    updateRibbonFromResult(result);
    updateTrialCounter(result.trial ?? result.step ?? 0);
  } catch (e) {
    if (token === cancelToken) console.error('Step failed:', e);
  } finally {
    if (token === cancelToken) { isRunning = false; setButtons(true); }
  }
}

async function doRun() {
  const token = cancelToken;
  isRunning = true;
  setButtons(false);
  try {
    const n = 10;
    for (let i = 0; i < n; i++) {
      if (token !== cancelToken) return;
      const result = callStep(pyodide);
      if (token !== cancelToken) return;
      await controller!.animateStep(result);
      if (token !== cancelToken) return;
      controller!.updatePanel(result);
      updateTrialCounter(result.trial ?? result.step ?? 0);
    }
  } finally {
    if (token === cancelToken) { isRunning = false; setButtons(true); }
  }
}

async function doExperiment() {
  const token = cancelToken;
  const nSteps = DEFAULT_STEPS[currentScenarioId] ?? 32;
  isRunning = true;
  setButtons(false);
  try {
    const results = callRunExperiment(pyodide, nSteps);
    for (const result of results) {
      if (token !== cancelToken) return;
      await controller!.animateStep(result);
      if (token !== cancelToken) return;
      controller!.updatePanel(result);
      updateTrialCounter(result.trial ?? result.step ?? 0);
    }
  } finally {
    if (token === cancelToken) { isRunning = false; setButtons(true); }
  }
}

async function doBenchmarkBatch() {
  const token = cancelToken;
  isRunning = true;
  setButtons(false);
  try {
    if (!experimentDashboard) return;
    const settings = experimentDashboard.getSettings();
    const completed = experimentDashboard.getCompletedEpisodes();
    if (completed >= settings.episodesPerAgent) {
      experimentDashboard.setStatus('Benchmark already complete. Clear results or increase episodes to continue.');
      return;
    }
    const batchEpisodes = Math.min(settings.batchSize, settings.episodesPerAgent - completed);
    const ablationTag = settings.forceExtrinsicOnly ? ' [EFE ablation]' : '';
    experimentDashboard.setStatus(`Running batch ${completed + 1}-${completed + batchEpisodes} / ${settings.episodesPerAgent} per agent${ablationTag}...`);
    const batch = await callBenchmarkBatchAsync({
      scenarioId: currentScenarioId,
      nSteps: settings.stepCap,
      episodesPerAgent: batchEpisodes,
      agentTypes: (AGENTS[currentScenarioId] ?? []).map(a => a.value),
      modelParams: getBenchmarkModelParams(),
      forceExtrinsicOnly: settings.forceExtrinsicOnly,
    });
    if (token !== cancelToken) return;
    experimentDashboard.applyBatch(batch);
    const updated = experimentDashboard.getCompletedEpisodes();
    experimentDashboard.setStatus(updated >= settings.episodesPerAgent
      ? `Benchmark complete — ${updated} episodes per agent.`
      : `Completed ${updated} / ${settings.episodesPerAgent} episodes per agent.`);
    updateTrialCounter(updated);
  } catch (error) {
    if (token !== cancelToken) return;
    experimentDashboard?.setStatus(error instanceof Error ? error.message : 'Benchmark failed.');
    console.error('Benchmark batch failed:', error);
  } finally {
    if (token === cancelToken) { isRunning = false; setButtons(true); }
  }
}

async function doBenchmark() {
  const token = cancelToken;
  isRunning = true;
  setButtons(false);
  try {
    if (!experimentDashboard) return;
    const settings = experimentDashboard.getSettings();
    while (experimentDashboard.getCompletedEpisodes() < settings.episodesPerAgent) {
      if (token !== cancelToken) return;
      const completed = experimentDashboard.getCompletedEpisodes();
      const batchEpisodes = Math.min(settings.batchSize, settings.episodesPerAgent - completed);
      const ablationTag = settings.forceExtrinsicOnly ? ' [EFE ablation]' : '';
      experimentDashboard.setStatus(`Running batch ${completed + 1}-${completed + batchEpisodes} / ${settings.episodesPerAgent} per agent${ablationTag}...`);
      const batch = await callBenchmarkBatchAsync({
        scenarioId: currentScenarioId,
        nSteps: settings.stepCap,
        episodesPerAgent: batchEpisodes,
        agentTypes: (AGENTS[currentScenarioId] ?? []).map(a => a.value),
        modelParams: getBenchmarkModelParams(),
        forceExtrinsicOnly: settings.forceExtrinsicOnly,
      });
      if (token !== cancelToken) return;
      experimentDashboard.applyBatch(batch);
      updateTrialCounter(experimentDashboard.getCompletedEpisodes());
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    }
    experimentDashboard.setStatus(`Benchmark complete — ${experimentDashboard.getCompletedEpisodes()} episodes per agent.`);
  } catch (error) {
    if (token !== cancelToken) return;
    experimentDashboard?.setStatus(error instanceof Error ? error.message : 'Benchmark failed.');
    console.error('Benchmark failed:', error);
  } finally {
    if (token === cancelToken) { isRunning = false; setButtons(true); }
  }
}

function wireDroneV2PanelControls() {
  const trainBtn = document.getElementById('v2-btn-train') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('v2-btn-reset-alpha') as HTMLButtonElement | null;
  if (!trainBtn || !resetBtn || !controller) return;
  const v2 = controller as any;

  trainBtn.addEventListener('click', async () => {
    if (isRunning) return;
    const token = cancelToken;
    isRunning = true;
    setButtons(false);
    trainBtn.disabled = true;
    resetBtn.disabled = true;
    const N_EPS = 10;
    const nSteps = DEFAULT_STEPS[currentScenarioId] ?? 400;
    if (typeof v2.setTrainStatus === 'function') v2.setTrainStatus(`Training ${N_EPS} episodes (α accumulating)...`);
    try {
      // Yield to the event loop so the status paints before the blocking run.
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      if (token !== cancelToken) return;
      const out = callTrainEpisodes(pyodide, currentAgent, N_EPS, nSteps);
      if (token !== cancelToken) return;
      const scores = (out.summaries ?? []).map((s: any) => s.reward ?? 0);
      if (typeof v2.appendTrainingScores === 'function') v2.appendTrainingScores(scores);
      if (typeof v2.updateAlphaDisplay === 'function' && out.world_alpha) {
        v2.updateAlphaDisplay(out.world_alpha);
      }
      const last10 = scores.slice(-10);
      const avg = last10.length ? last10.reduce((a: number, b: number) => a + b, 0) / last10.length : 0;
      if (typeof v2.setTrainStatus === 'function') v2.setTrainStatus(`Trained ${N_EPS} eps. Last-10 mean score: ${avg.toFixed(2)}. α continues accumulating.`);
      // Soft-reset the scene so the next Step starts from a fresh episode.
      callReset(pyodide, currentAgent);
      if (controller) { controller.reset(); }
    } catch (e) {
      console.error('Train failed:', e);
      if (typeof v2.setTrainStatus === 'function') v2.setTrainStatus('Training failed — see console.');
    } finally {
      if (token === cancelToken) {
        isRunning = false;
        setButtons(true);
        trainBtn.disabled = false;
        resetBtn.disabled = false;
      }
    }
  });

  resetBtn.addEventListener('click', () => {
    if (isRunning) return;
    callHardReset(pyodide, currentAgent);
    if (controller) { controller.reset(); controller.resetPanel(); }
    if (typeof v2.clearTrainingCurve === 'function') v2.clearTrainingCurve();
    if (typeof v2.updateAlphaDisplay === 'function') v2.updateAlphaDisplay([1, 1, 1, 1]);
    if (typeof v2.setTrainStatus === 'function') v2.setTrainStatus('α reset to uniform prior. Drone starts from scratch.');
  });
}

function clearBenchmarkResults() {
  if (!experimentDashboard) return;
  experimentDashboard.resetResults();
  experimentDashboard.setStatus('Results cleared. Ready to benchmark all agents.');
  updateTrialCounter(0);
}

function setButtons(enabled: boolean) {
  for (const id of ['btn-step', 'btn-run', 'btn-experiment', 'btn-reset']) {
    (document.getElementById(id) as HTMLButtonElement).disabled = !enabled;
  }
}

function updateTrialCounter(n: number) {
  const el = document.getElementById('trial-num');
  if (el) el.textContent = String(n);
}

type RibbonInfo = { policy: string; policyColor: string; driver: string; driverColor: string } | null;
function setRibbon(info: RibbonInfo, _result: any) {
  const ribbon = document.getElementById('viewport-ribbon');
  const pEl = document.getElementById('rb-policy');
  const dEl = document.getElementById('rb-driver');
  if (!ribbon || !pEl || !dEl) return;
  if (!info) {
    ribbon.classList.remove('visible');
    pEl.textContent = '—'; pEl.style.color = '';
    dEl.textContent = '—'; dEl.style.color = '';
    return;
  }
  pEl.textContent = info.policy;
  pEl.style.color = info.policyColor;
  dEl.textContent = info.driver;
  dEl.style.color = info.driverColor;
  ribbon.classList.add('visible');
}

const COMP_COLORS: Record<string, string> = {
  extrinsic: '#22c55e', salience: '#3b82f6', novelty: '#f59e0b',
};
const COMP_LABELS: Record<string, string> = {
  extrinsic: 'Extrinsic value', salience: 'Salience', novelty: 'Novelty',
};

function updateRibbonFromResult(result: any) {
  // Policy label + color vary by scenario. Prefer explicit fields.
  let policyName = '—';
  let policyColor = '#d8d8ee';
  let efeEntry: Record<string, number> | undefined;

  if (result.policy && result.efe?.[result.policy]) {
    // T-maze
    const pMap: Record<string, string> = {
      left_direct: 'L direct', right_direct: 'R direct', cue_then_best: 'Cue → Best',
    };
    const cMap: Record<string, string> = {
      left_direct: '#22c55e', right_direct: '#f59e0b', cue_then_best: '#3b82f6',
    };
    policyName = pMap[result.policy] ?? result.policy;
    policyColor = cMap[result.policy] ?? '#d8d8ee';
    efeEntry = result.efe[result.policy];
  } else if (result.waypoint && result.efe?.[result.waypoint]) {
    // Drone
    policyName = result.waypoint;
    policyColor = '#06b6d4';
    efeEntry = result.efe[result.waypoint];
  } else if (result.goal && result.goal_efe?.[result.goal]) {
    // Grid maze
    policyName = result.goal;
    policyColor = result.goal_type === 'informant' ? '#06b6d4' : '#f8fafc';
    const g = result.goal_efe[result.goal];
    efeEntry = { extrinsic: g.extrinsic, salience: g.salience, novelty: g.novelty };
  }

  if (!efeEntry) {
    setRibbon(null, result);
    return;
  }

  const vals: [string, number][] = [
    ['extrinsic', Math.abs(efeEntry.extrinsic ?? 0)],
    ['salience',  Math.abs(efeEntry.salience  ?? 0)],
    ['novelty',   Math.abs(efeEntry.novelty   ?? 0)],
  ];
  vals.sort((a, b) => b[1] - a[1]);
  const [topKey, topVal] = vals[0];
  const tied = vals[1][1] >= topVal * 0.8 && topVal > 0;
  const driver = tied
    ? `${COMP_LABELS[topKey]} ≈ ${COMP_LABELS[vals[1][0]]}`
    : `${COMP_LABELS[topKey]} dominates`;
  const driverColor = COMP_COLORS[topKey] ?? '#d8d8ee';

  setRibbon({ policy: policyName, policyColor, driver, driverColor }, result);
}

function onResize() {
  const viewport = document.getElementById('viewport')!;
  if (sceneObjects) {
    const w = viewport.clientWidth;
    const h = viewport.clientHeight;
    sceneObjects.camera.aspect = w / h;
    sceneObjects.camera.updateProjectionMatrix();
    sceneObjects.renderer.setSize(w, h);
  }
  if (experimentDashboard) experimentDashboard.resize();
}

boot().catch(err => {
  status.textContent = `Error: ${err.message}`;
  console.error(err);
});
