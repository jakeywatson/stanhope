import { callStep, callReset, callRunExperiment } from '../bridge';
import { AgentAvatar } from '../scene/AgentAvatar';
import * as THREE from 'three';

type StepResult = any;

const AGENT_DESCRIPTIONS: Record<string, string> = {
  active_learning:
    '<strong>Active Learning</strong> — explores to reduce uncertainty about reward probabilities (parameter exploration). Visits uncertain arms to learn their payoff mappings.',
  active_inference:
    '<strong>Active Inference</strong> — explores to infer the current hidden context (hidden-state exploration). Visits the cue to learn which arm is rewarding this trial.',
  combined:
    '<strong>Combined</strong> — both hidden-state and parameter exploration. Visits the cue for context AND uncertain arms for learning.',
  greedy:
    '<strong>Greedy</strong> — pure reward maximisation. Goes directly to whichever arm has highest expected value. No exploration.',
  random:
    '<strong>Random</strong> — softmax with low precision. Explores by accident, not by design.',
};

const POLICY_LABELS: Record<string, string> = {
  left_direct: 'L direct',
  right_direct: 'R direct',
  cue_then_best: 'Cue→Best',
};
const POLICY_COLORS: Record<string, string> = {
  left_direct: '#22c55e',
  right_direct: '#f59e0b',
  cue_then_best: '#3b82f6',
};
const POLICY_ORDER = ['left_direct', 'right_direct', 'cue_then_best'];

let currentAgent = 'active_learning';
let isRunning = false;
let rewardHistory: number[] = [];

export function setupUI(pyodide: any, avatar: AgentAvatar, scene: THREE.Scene) {
  const agentSelect = document.getElementById('agent-select') as HTMLSelectElement;
  const btnStep = document.getElementById('btn-step') as HTMLButtonElement;
  const btnRun = document.getElementById('btn-run') as HTMLButtonElement;
  const btnExperiment = document.getElementById('btn-experiment') as HTMLButtonElement;
  const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  const trialNum = document.getElementById('trial-num') as HTMLSpanElement;
  const agentInfo = document.getElementById('agent-info') as HTMLDivElement;

  callReset(pyodide, currentAgent);

  agentSelect.addEventListener('change', () => {
    currentAgent = agentSelect.value;
    agentInfo.innerHTML = AGENT_DESCRIPTIONS[currentAgent] || '';
    callReset(pyodide, currentAgent);
    avatar.resetToStart();
    trialNum.textContent = '0';
    rewardHistory = [];
    resetUI();
  });

  btnStep.addEventListener('click', async () => {
    if (isRunning) return;
    isRunning = true;
    const result = callStep(pyodide);
    await animateStep(result, avatar);
    updateUI(result);
    isRunning = false;
  });

  btnRun.addEventListener('click', async () => {
    if (isRunning) return;
    isRunning = true;
    const result = callStep(pyodide);
    await animateStep(result, avatar);
    updateUI(result);
    isRunning = false;
  });

  btnExperiment.addEventListener('click', async () => {
    if (isRunning) return;
    isRunning = true;
    const results = callRunExperiment(pyodide, 32);
    for (const result of results) {
      await animateStep(result, avatar);
      updateUI(result);
      await sleep(200);
    }
    isRunning = false;
  });

  btnReset.addEventListener('click', () => {
    callReset(pyodide, currentAgent);
    avatar.resetToStart();
    trialNum.textContent = '0';
    rewardHistory = [];
    resetUI();
  });
}

async function animateStep(result: StepResult, avatar: AgentAvatar) {
  // Walk the trajectory: the agent always starts at centre
  for (const step of result.trajectory) {
    await avatar.moveToLocation(step.location);
    await sleep(250);
  }
  // Flash reward feedback
  flashReward(result.trajectory[result.trajectory.length - 1]?.observation ?? '');
  await sleep(350);
  // Return to centre
  avatar.resetToStart();
}

function updateUI(result: StepResult) {
  const trialNum = document.getElementById('trial-num') as HTMLSpanElement;
  trialNum.textContent = String(result.trial);

  // Hidden context indicator
  const ctxEl = document.getElementById('context-indicator') as HTMLElement;
  if (ctxEl) {
    ctxEl.innerHTML = result.context === 'left_good'
      ? '<span style="color:#22c55e">&#9679;</span> Left rewarding'
      : '<span style="color:#f59e0b">&#9679;</span> Right rewarding';
  }

  // Context belief
  setBar('bar-ctx-left', 'val-ctx-left', result.beliefs.context_belief[0]);
  setBar('bar-ctx-right', 'val-ctx-right', result.beliefs.context_belief[1]);

  // Left arm
  setBar('bar-left-p', 'val-left-p', result.beliefs.left_arm.p_reward);
  setConc('bar-left-conc-r', 'val-left-conc-r', result.beliefs.left_arm.conc_reward);
  setConc('bar-left-conc-l', 'val-left-conc-l', result.beliefs.left_arm.conc_loss);

  // Right arm
  setBar('bar-right-p', 'val-right-p', result.beliefs.right_arm.p_reward);
  setConc('bar-right-conc-r', 'val-right-conc-r', result.beliefs.right_arm.conc_reward);
  setConc('bar-right-conc-l', 'val-right-conc-l', result.beliefs.right_arm.conc_loss);

  // Policy probabilities
  for (const pName of POLICY_ORDER) {
    const el = document.getElementById(`pol-${pName}`) as HTMLDivElement | null;
    if (el) el.style.width = `${(result.policy_probs[pName] ?? 0) * 100}%`;
    const vEl = document.getElementById(`pol-val-${pName}`) as HTMLSpanElement | null;
    if (vEl) vEl.textContent = (result.policy_probs[pName] ?? 0).toFixed(2);
  }

  // EFE decomposition
  for (const pName of POLICY_ORDER) {
    const efe = result.efe[pName];
    if (!efe) continue;
    setEFE(`efe-${pName}-ext`, efe.extrinsic);
    setEFE(`efe-${pName}-sal`, efe.salience);
    setEFE(`efe-${pName}-nov`, efe.novelty);
  }

  // Chosen policy highlight
  const chosenEl = document.getElementById('chosen-policy') as HTMLElement | null;
  if (chosenEl) {
    const label = POLICY_LABELS[result.policy] ?? result.policy;
    const color = POLICY_COLORS[result.policy] ?? '#fff';
    chosenEl.innerHTML = `<span style="color:${color}">${label}</span>`;
  }

  // Reward chart
  rewardHistory.push(result.reward);
  drawRewardChart();
}

function setBar(barId: string, valId: string, value: number) {
  const bar = document.getElementById(barId) as HTMLDivElement | null;
  const val = document.getElementById(valId) as HTMLSpanElement | null;
  if (bar) bar.style.width = `${value * 100}%`;
  if (val) val.textContent = value.toFixed(2);
}

function setConc(barId: string, valId: string, value: number) {
  const bar = document.getElementById(barId) as HTMLDivElement | null;
  const val = document.getElementById(valId) as HTMLSpanElement | null;
  const pct = Math.min(value / 20, 1) * 100;
  if (bar) bar.style.width = `${pct}%`;
  if (val) val.textContent = value.toFixed(1);
}

function setEFE(id: string, value: number) {
  const el = document.getElementById(id) as HTMLDivElement | null;
  if (!el) return;
  const maxVal = 5;
  el.style.width = `${Math.min(Math.abs(value) / maxVal, 1) * 100}%`;
}

function resetUI() {
  setBar('bar-ctx-left', 'val-ctx-left', 0.5);
  setBar('bar-ctx-right', 'val-ctx-right', 0.5);
  setBar('bar-left-p', 'val-left-p', 0.5);
  setBar('bar-right-p', 'val-right-p', 0.5);
  setConc('bar-left-conc-r', 'val-left-conc-r', 1.0);
  setConc('bar-left-conc-l', 'val-left-conc-l', 1.0);
  setConc('bar-right-conc-r', 'val-right-conc-r', 1.0);
  setConc('bar-right-conc-l', 'val-right-conc-l', 1.0);
  for (const pName of POLICY_ORDER) {
    setEFE(`efe-${pName}-ext`, 0);
    setEFE(`efe-${pName}-sal`, 0);
    setEFE(`efe-${pName}-nov`, 0);
    const el = document.getElementById(`pol-${pName}`) as HTMLDivElement | null;
    if (el) el.style.width = `${100 / POLICY_ORDER.length}%`;
    const vEl = document.getElementById(`pol-val-${pName}`) as HTMLSpanElement | null;
    if (vEl) vEl.textContent = (1 / POLICY_ORDER.length).toFixed(2);
  }
  const ctxEl = document.getElementById('context-indicator') as HTMLElement;
  if (ctxEl) ctxEl.innerHTML = '—';
  const chosenEl = document.getElementById('chosen-policy') as HTMLElement | null;
  if (chosenEl) chosenEl.innerHTML = '—';
}

function drawRewardChart() {
  const canvas = document.getElementById('reward-chart') as HTMLCanvasElement;
  const container = document.getElementById('reward-chart-container') as HTMLDivElement;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (rewardHistory.length === 0) return;

  const cumulative: number[] = [];
  let sum = 0;
  for (const r of rewardHistory) {
    sum += r;
    cumulative.push(sum);
  }

  const maxTrials = Math.max(32, rewardHistory.length);
  const maxReward = Math.max(Math.abs(sum), 1);
  const padX = 30;
  const padY = 10;
  const plotW = canvas.width - padX - 10;
  const plotH = canvas.height - padY * 2;

  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, padY);
  ctx.lineTo(padX, padY + plotH);
  ctx.lineTo(padX + plotW, padY + plotH);
  ctx.stroke();

  ctx.strokeStyle = '#7c3aed';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < cumulative.length; i++) {
    const x = padX + (i / maxTrials) * plotW;
    const y = padY + plotH - (cumulative[i] / maxReward) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#7c3aed';
  for (let i = 0; i < cumulative.length; i++) {
    const x = padX + (i / maxTrials) * plotW;
    const y = padY + plotH - (cumulative[i] / maxReward) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function flashReward(observation: string) {
  const viewport = document.getElementById('viewport') as HTMLDivElement;
  const flash = document.createElement('div');
  const bg = observation === 'reward'
    ? 'rgba(139,92,246,0.15)'
    : observation === 'loss'
      ? 'rgba(239,68,68,0.1)'
      : 'rgba(59,130,246,0.08)';
  flash.style.cssText = `position:absolute;inset:0;pointer-events:none;background:${bg};transition:opacity 0.5s;`;
  viewport.appendChild(flash);
  setTimeout(() => { flash.style.opacity = '0'; }, 50);
  setTimeout(() => flash.remove(), 600);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
