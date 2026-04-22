import * as THREE from 'three';
import {
  type SceneController, type SceneObjects,
  createBaseRenderer, addBaseLighting, sleep, annotateEFE,
} from './types';

// ─── Maze positions and paths ───

const CENTRE = new THREE.Vector3(0, 0, 0);
const CUE = new THREE.Vector3(0, 0, -10);
const LEFT = new THREE.Vector3(-10, 0, 0);
const RIGHT = new THREE.Vector3(10, 0, 0);
const START = new THREE.Vector3(0, 0, 7);

const POSITIONS: Record<string, THREE.Vector3> = {
  centre: CENTRE, cue: CUE, left: LEFT, right: RIGHT, start: START,
};

const POLICY_ORDER = ['left_direct', 'right_direct', 'cue_then_best'];
const POLICY_LABELS: Record<string, string> = {
  left_direct: 'L direct', right_direct: 'R direct',
  cue_then_best: 'Cue→Best',
};
const POLICY_COLORS: Record<string, string> = {
  left_direct: '#22c55e', right_direct: '#f59e0b',
  cue_then_best: '#3b82f6',
};

export class TMazeScene implements SceneController {
  private sceneObj!: THREE.Scene;
  private agent!: THREE.Mesh;
  private glowLight!: THREE.PointLight;
  private waypoints: THREE.Vector3[] = [];
  private currentTarget!: THREE.Vector3;
  private moving = false;
  private onArrive: (() => void) | null = null;
  private rewardHistory: number[] = [];
  private trailSegments: THREE.Line[] = [];
  private trailPointsCurrent: THREE.Vector3[] = [];
  private ghostSegments: THREE.Line[] = [];

  init(container: HTMLElement): SceneObjects {
    const { renderer, scene, camera, controls } = createBaseRenderer(container);
    this.sceneObj = scene;

    camera.position.set(18, 22, 22);
    camera.lookAt(0, 0, -1);
    controls.target.set(0, 0, -1);

    addBaseLighting(scene);

    // Ground
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(80, 80),
      new THREE.MeshStandardMaterial({ color: 0x0c0c18, roughness: 0.95 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.1;
    ground.receiveShadow = true;
    scene.add(ground);
    scene.add(new THREE.GridHelper(60, 60, 0x151528, 0x151528));

    this.buildMaze(scene);
    this.createAgent(scene);

    return {
      scene, camera, renderer, controls,
      onFrame: () => this.updateAgent(),
    };
  }

  async animateStep(result: any): Promise<void> {
    // Fade the previous trial's trail into a ghost before starting the new one.
    this.demoteTrailToGhost();
    this.trailPointsCurrent = [POSITIONS.centre.clone().setY(0.15)];

    const policyColor = this.policyColorForResult(result);
    const traj = result.trajectory;
    for (let i = 0; i < traj.length; i++) {
      const loc = traj[i].location;
      const prevLoc = i > 0 ? traj[i - 1].location : null;
      if (loc === prevLoc) {
        await sleep(250);
        continue;
      }
      if (prevLoc && prevLoc !== 'centre' && loc !== 'centre') {
        await this.moveToLocation('centre');
        this.extendTrail('centre', policyColor);
        await sleep(150);
      }
      await this.moveToLocation(loc);
      this.extendTrail(loc, policyColor);
      await sleep(250);
    }
    this.flashReward(traj[traj.length - 1]?.observation ?? '');
    await sleep(350);
    this.resetAgent();
  }

  private policyColorForResult(result: any): number {
    const map: Record<string, number> = {
      left_direct: 0x22c55e, right_direct: 0xf59e0b, cue_then_best: 0x3b82f6,
    };
    return map[result.policy] ?? 0x7c3aed;
  }

  private extendTrail(location: string, colorHex: number) {
    const dest = POSITIONS[location];
    if (!dest) return;
    const pt = dest.clone(); pt.y = 0.15;
    this.trailPointsCurrent.push(pt);
    if (this.trailPointsCurrent.length < 2) return;
    const pts = this.trailPointsCurrent.slice(-2);
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.7 });
    const line = new THREE.Line(geo, mat);
    this.sceneObj.add(line);
    this.trailSegments.push(line);
  }

  private demoteTrailToGhost() {
    for (const seg of this.trailSegments) {
      const mat = seg.material as THREE.LineBasicMaterial;
      mat.opacity = 0.15;
      this.ghostSegments.push(seg);
    }
    this.trailSegments = [];
    // Cap ghost trail length so it doesn't accumulate forever.
    while (this.ghostSegments.length > 40) {
      const old = this.ghostSegments.shift();
      if (old) this.sceneObj.remove(old);
    }
  }

  reset(): void {
    this.resetAgent();
    this.rewardHistory = [];
    for (const seg of this.trailSegments) this.sceneObj.remove(seg);
    this.trailSegments = [];
    for (const seg of this.ghostSegments) this.sceneObj.remove(seg);
    this.ghostSegments = [];
    this.trailPointsCurrent = [];
  }

  dispose(): void {
    // Three.js cleanup handled by main.ts
  }

  // ─── Panel ───

  buildPanel(): string {
    return `
      <details class="theory-card">
        <summary>Paper Connection — Schwartenbeck et al. 2019</summary>
        <div class="tc-body">
          <p>The T-maze is the <strong>core paradigm</strong> from the paper. The agent must decide whether to go directly to an arm (gamble) or first visit the <strong>cue</strong> location to resolve uncertainty.</p>
          <div class="tc-eq">G(π) = <span class="tc-ext">Extrinsic value</span> + <span class="tc-sal">Salience</span> + <span class="tc-nov">Novelty</span></div>
          <p><span class="tc-label tc-ext">Extrinsic</span> — expected reward from choosing an arm</p>
          <p><span class="tc-label tc-sal">Salience</span> — information gain about which arm is rewarding (drives cue-seeking)</p>
          <p><span class="tc-label tc-nov">Novelty</span> — parameter learning about the risky arm's big/none concentrations (Dirichlet α)</p>
          <p>Watch: the <em>Cue→Best</em> plan has high salience early on, driving the agent to inspect the cue and then replan with updated context beliefs.</p>
        </div>
      </details>
      <div class="efe-annotation" id="efe-annotation">
        <div class="ann-driver" id="ann-driver">Awaiting first step...</div>
        <div id="ann-detail"></div>
      </div>
      <div class="panel-section">
        <h3>Hidden Context</h3>
        <div style="font-size:0.75rem;color:#8080a0;margin-bottom:0.4rem;">True: <span id="context-indicator">—</span></div>
        <div class="belief-bar"><label>P(risky good)</label><div class="bar-track"><div class="bar-fill fill-safe" id="bar-ctx-left" style="width:50%"></div></div><span class="bar-value" id="val-ctx-left">0.50</span></div>
        <div class="belief-bar"><label>P(risky bad)</label><div class="bar-track"><div class="bar-fill fill-risky" id="bar-ctx-right" style="width:50%"></div></div><span class="bar-value" id="val-ctx-right">0.50</span></div>
      </div>
      <div class="panel-section">
        <h3>Arm Beliefs (Dirichlet)</h3>
        <div style="font-size:0.7rem;color:#22c55e;margin-bottom:0.25rem;">Left arm (safe)</div>
        <div class="belief-bar"><label>P(small)</label><div class="bar-track"><div class="bar-fill fill-safe" id="bar-left-p" style="width:100%"></div></div><span class="bar-value" id="val-left-p">1.00</span></div>
        <div style="font-size:0.65rem;color:#666680;font-style:italic;margin:0.1rem 0 0.3rem;">deterministic — not learned</div>
        <div style="font-size:0.7rem;color:#f59e0b;margin:0.4rem 0 0.25rem;">Right arm (risky)</div>
        <div class="belief-bar"><label>P(big)</label><div class="bar-track"><div class="bar-fill fill-risky" id="bar-right-p" style="width:50%"></div></div><span class="bar-value" id="val-right-p">0.50</span></div>
        <div class="belief-bar"><label>α big</label><div class="bar-track"><div class="bar-fill fill-novelty" id="bar-right-conc-r" style="width:10%"></div></div><span class="bar-value" id="val-right-conc-r">1.0</span></div>
        <div class="belief-bar"><label>α none</label><div class="bar-track"><div class="bar-fill fill-reward-none" id="bar-right-conc-l" style="width:10%"></div></div><span class="bar-value" id="val-right-conc-l">1.0</span></div>
      </div>
      <div class="panel-section">
        <h3>Policy Probabilities</h3>
        <div style="font-size:0.7rem;color:#8080a0;margin-bottom:0.3rem;">Chosen: <span id="chosen-policy">—</span></div>
        ${POLICY_ORDER.map(p => `
          <div class="belief-bar">
            <label style="color:${POLICY_COLORS[p]}">${POLICY_LABELS[p]}</label>
            <div class="bar-track"><div class="bar-fill" id="pol-${p}" style="width:${100 / POLICY_ORDER.length}%;background:${POLICY_COLORS[p]}"></div></div>
            <span class="bar-value" id="pol-val-${p}">${(1 / POLICY_ORDER.length).toFixed(2)}</span>
          </div>
        `).join('')}
      </div>
      <div class="panel-section">
        <h3>Expected Free Energy</h3>
        ${POLICY_ORDER.map(p => `
          <div style="font-size:0.65rem;color:${POLICY_COLORS[p]};margin:0.3rem 0 0.2rem;">${POLICY_LABELS[p]}</div>
          <div class="efe-row"><label>Extrinsic</label><div class="efe-bar"><div class="efe-fill fill-extrinsic" id="efe-${p}-ext" style="width:0%"></div></div></div>
          <div class="efe-row"><label>Salience</label><div class="efe-bar"><div class="efe-fill fill-salience" id="efe-${p}-sal" style="width:0%"></div></div></div>
          <div class="efe-row"><label>Novelty</label><div class="efe-bar"><div class="efe-fill fill-novelty" id="efe-${p}-nov" style="width:0%"></div></div></div>
        `).join('')}
      </div>
      <div class="panel-section">
        <h3>Cumulative Reward</h3>
        <div id="reward-chart-container"><canvas id="reward-chart"></canvas></div>
      </div>
    `;
  }

  updatePanel(result: any): void {
    const el = (id: string) => document.getElementById(id);

    // Context
    const ctx = el('context-indicator');
    if (ctx) ctx.innerHTML = result.context === 'risky_good'
      ? '<span style="color:#f59e0b">●</span> Risky arm paying big'
      : '<span style="color:#22c55e">●</span> Risky arm paying nothing';

    // Context belief
    setBar('bar-ctx-left', 'val-ctx-left', result.beliefs.context_belief[0]);
    setBar('bar-ctx-right', 'val-ctx-right', result.beliefs.context_belief[1]);

    // Arms
    setBar('bar-left-p', 'val-left-p', result.beliefs.left_arm.p_reward);
    setBar('bar-right-p', 'val-right-p', result.beliefs.right_arm.p_reward);
    setConc('bar-right-conc-r', 'val-right-conc-r', result.beliefs.right_arm.conc_reward);
    setConc('bar-right-conc-l', 'val-right-conc-l', result.beliefs.right_arm.conc_loss);

    // Policy probs
    for (const p of POLICY_ORDER) {
      const pe = el(`pol-${p}`) as HTMLDivElement | null;
      if (pe) pe.style.width = `${(result.policy_probs[p] ?? 0) * 100}%`;
      const ve = el(`pol-val-${p}`);
      if (ve) ve.textContent = (result.policy_probs[p] ?? 0).toFixed(2);
    }

    // EFE
    for (const p of POLICY_ORDER) {
      const efe = result.efe[p];
      if (!efe) continue;
      setEFE(`efe-${p}-ext`, efe.extrinsic);
      setEFE(`efe-${p}-sal`, efe.salience);
      setEFE(`efe-${p}-nov`, efe.novelty);
    }

    // Chosen policy
    const cp = el('chosen-policy');
    if (cp) {
      const label = POLICY_LABELS[result.policy] ?? result.policy;
      const color = POLICY_COLORS[result.policy] ?? '#fff';
      cp.innerHTML = `<span style="color:${color}">${label}</span>`;
    }

    // Reward
    this.rewardHistory.push(result.reward);
    drawChart(this.rewardHistory);

    // Live annotation
    annotateEFE(result.efe, result.policy, {
      extrinsic: 'The agent expects reward from this arm — going direct.',
      salience: 'The agent seeks information — visiting the cue to learn which arm is rewarding.',
      novelty: 'The agent is learning reward concentrations — updating its Dirichlet parameters.',
      tie: 'Multiple drives are competing — the agent balances information-seeking with reward.',
    });
  }

  resetPanel(): void {
    this.rewardHistory = [];
    setBar('bar-ctx-left', 'val-ctx-left', 0.5);
    setBar('bar-ctx-right', 'val-ctx-right', 0.5);
    setBar('bar-left-p', 'val-left-p', 1.0);
    setBar('bar-right-p', 'val-right-p', 0.8);
    setConc('bar-right-conc-r', 'val-right-conc-r', 4.0);
    setConc('bar-right-conc-l', 'val-right-conc-l', 1.0);
    for (const p of POLICY_ORDER) {
      setEFE(`efe-${p}-ext`, 0);
      setEFE(`efe-${p}-sal`, 0);
      setEFE(`efe-${p}-nov`, 0);
      const pe = document.getElementById(`pol-${p}`) as HTMLDivElement | null;
      if (pe) pe.style.width = `${100 / POLICY_ORDER.length}%`;
      const ve = document.getElementById(`pol-val-${p}`);
      if (ve) ve.textContent = (1 / POLICY_ORDER.length).toFixed(2);
    }
    const ctx = document.getElementById('context-indicator');
    if (ctx) ctx.innerHTML = '—';
    const cp = document.getElementById('chosen-policy');
    if (cp) cp.innerHTML = '—';
  }

  // ─── Maze building ───

  private buildMaze(scene: THREE.Scene) {
    const fm = new THREE.MeshStandardMaterial({ color: 0x181830, roughness: 0.85 });
    const wm = new THREE.MeshStandardMaterial({ color: 0x28284a, roughness: 0.5, metalness: 0.15 });
    const CW = 3; const WH = 2.0;

    // Floors
    addBox(scene, fm, CW, 0.12, 8, 0, -0.04, -5);
    addBox(scene, fm, CW, 0.12, 5, 0, -0.04, 3.5);
    addBox(scene, fm, 8, 0.12, CW, -5, -0.04, 0);
    addBox(scene, fm, 8, 0.12, CW, 5, -0.04, 0);
    addBox(scene, fm, CW + 2, 0.12, CW + 2, 0, -0.04, 0);

    // Room platforms
    const cueMat = makeMat(0x3b82f6);
    const leftMat = makeMat(0x22c55e);
    const rightMat = makeMat(0xf59e0b);
    const startMat = makeMat(0x7c3aed);
    addBox(scene, cueMat, 3.5, 0.18, 3.5, 0, 0.02, -10);
    addBox(scene, leftMat, 3.5, 0.18, 3.5, -10, 0.02, 0);
    addBox(scene, rightMat, 3.5, 0.18, 3.5, 10, 0.02, 0);
    addBox(scene, startMat, 3, 0.18, 3, 0, 0.02, 7);

    // Walls (corridors + rooms) — abbreviated for clarity
    const WT = 0.18;
    // North corridor
    addBox(scene, wm, WT, WH, 8, -CW / 2, WH / 2, -5);
    addBox(scene, wm, WT, WH, 8, CW / 2, WH / 2, -5);
    addBox(scene, wm, 4, WH, WT, 0, WH / 2, -11.75);
    addBox(scene, wm, WT, WH, 3.5, -2, WH / 2, -10);
    addBox(scene, wm, WT, WH, 3.5, 2, WH / 2, -10);
    // South corridor
    addBox(scene, wm, WT, WH, 5, -CW / 2, WH / 2, 3.5);
    addBox(scene, wm, WT, WH, 5, CW / 2, WH / 2, 3.5);
    addBox(scene, wm, 3, WH, WT, 0, WH / 2, 8.5);
    addBox(scene, wm, WT, WH, 3, -1.5, WH / 2, 7);
    addBox(scene, wm, WT, WH, 3, 1.5, WH / 2, 7);
    // West
    addBox(scene, wm, 8, WH, WT, -5, WH / 2, -CW / 2);
    addBox(scene, wm, 8, WH, WT, -5, WH / 2, CW / 2);
    addBox(scene, wm, WT, WH, 4, -11.75, WH / 2, 0);
    addBox(scene, wm, 3.5, WH, WT, -10, WH / 2, -2);
    addBox(scene, wm, 3.5, WH, WT, -10, WH / 2, 2);
    // East
    addBox(scene, wm, 8, WH, WT, 5, WH / 2, -CW / 2);
    addBox(scene, wm, 8, WH, WT, 5, WH / 2, CW / 2);
    addBox(scene, wm, WT, WH, 4, 11.75, WH / 2, 0);
    addBox(scene, wm, 3.5, WH, WT, 10, WH / 2, -2);
    addBox(scene, wm, 3.5, WH, WT, 10, WH / 2, 2);

    // Features
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.5, 3, 8), cueMat);
    tower.position.set(0, 1.5, -10);
    tower.castShadow = true;
    scene.add(tower);
    scene.add(new THREE.PointLight(0x3b82f6, 3, 8).translateY(3.2).translateZ(-10));

    addPedestal(scene, leftMat, -10, 0);
    scene.add(new THREE.PointLight(0x22c55e, 2, 6).translateX(-10).translateY(2.5));
    addPedestal(scene, rightMat, 10, 0);
    scene.add(new THREE.PointLight(0xf59e0b, 2, 6).translateX(10).translateY(2.5));

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.7, 0.95, 32),
      new THREE.MeshBasicMaterial({ color: 0x7c3aed, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(0, 0.03, 7);
    scene.add(ring);

    addLabel(scene, 'CUE', 0, 3.8, -10, 0x3b82f6);
    addLabel(scene, 'LEFT', -10, 3.2, 0, 0x22c55e);
    addLabel(scene, 'RIGHT', 10, 3.2, 0, 0xf59e0b);
    addLabel(scene, 'START', 0, 2.8, 7, 0x7c3aed);
  }

  // ─── Agent ───

  private createAgent(scene: THREE.Scene) {
    const geo = new THREE.SphereGeometry(0.4, 32, 32);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x7c3aed, emissive: 0x7c3aed, emissiveIntensity: 0.5,
      roughness: 0.2, metalness: 0.6,
    });
    this.agent = new THREE.Mesh(geo, mat);
    this.agent.position.copy(POSITIONS.centre);
    this.agent.position.y = 0.5;
    this.agent.castShadow = true;
    scene.add(this.agent);

    this.glowLight = new THREE.PointLight(0x7c3aed, 2, 5);
    this.glowLight.position.copy(this.agent.position);
    scene.add(this.glowLight);

    this.currentTarget = this.agent.position.clone();
  }

  private moveToLocation(location: string): Promise<void> {
    return new Promise(resolve => {
      const dest = POSITIONS[location];
      if (!dest) { resolve(); return; }
      const target = dest.clone(); target.y = 0.5;
      // Midpoint between current position and destination (at y=0.5)
      const cur = this.agent.position.clone();
      const mid = cur.clone().lerp(target, 0.5); mid.y = 0.5;
      this.waypoints = [mid, target];
      this.currentTarget = this.waypoints.shift()!;
      this.moving = true;
      this.onArrive = resolve;
    });
  }

  private resetAgent() {
    const c = POSITIONS.centre.clone(); c.y = 0.5;
    this.agent.position.copy(c);
    this.currentTarget.copy(c);
    this.waypoints = [];
    this.moving = false;
    this.glowLight.position.copy(c);
  }

  private updateAgent() {
    if (this.moving) {
      const delta = this.currentTarget.clone().sub(this.agent.position);
      const dist = delta.length();
      if (dist < 0.1) {
        this.agent.position.copy(this.currentTarget);
        if (this.waypoints.length > 0) {
          this.currentTarget = this.waypoints.shift()!;
        } else {
          this.moving = false;
          const cb = this.onArrive; this.onArrive = null; cb?.();
        }
      } else {
        this.agent.position.add(delta.normalize().multiplyScalar(Math.min(0.14, dist)));
      }
    }
    this.agent.position.y = 0.5 + Math.sin(Date.now() * 0.003) * 0.05;
    this.glowLight.position.copy(this.agent.position);
  }

  private flashReward(observation: string) {
    const viewport = document.getElementById('viewport')!;
    const bg = observation === 'big' ? 'rgba(139,92,246,0.20)'
      : observation === 'small' ? 'rgba(34,197,94,0.15)'
      : observation === 'none' ? 'rgba(239,68,68,0.10)'
      : 'rgba(59,130,246,0.08)';
    const flash = document.createElement('div');
    flash.style.cssText = `position:absolute;inset:0;pointer-events:none;background:${bg};transition:opacity 0.5s;`;
    viewport.appendChild(flash);
    setTimeout(() => { flash.style.opacity = '0'; }, 50);
    setTimeout(() => flash.remove(), 600);
  }
}

// ─── Helpers ───

function addBox(scene: THREE.Scene, mat: THREE.Material, w: number, h: number, d: number, x: number, y: number, z: number) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true;
  scene.add(m);
}

function addPedestal(scene: THREE.Scene, mat: THREE.Material, x: number, z: number) {
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.7, 0.8, 12), mat);
  base.position.set(x, 0.4, z); base.castShadow = true; scene.add(base);
  const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.35, 0), mat);
  gem.position.set(x, 1.2, z); gem.castShadow = true; scene.add(gem);
}

function addLabel(scene: THREE.Scene, text: string, x: number, y: number, z: number, color: number) {
  const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
  ctx.font = 'bold 32px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true }),
  );
  sprite.position.set(x, y, z); sprite.scale.set(3, 0.75, 1);
  scene.add(sprite);
}

function makeMat(color: number) {
  return new THREE.MeshStandardMaterial({
    color, roughness: 0.35, metalness: 0.3, emissive: color, emissiveIntensity: 0.15,
  });
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
  if (bar) bar.style.width = `${Math.min(value / 20, 1) * 100}%`;
  if (val) val.textContent = value.toFixed(1);
}

function setEFE(id: string, value: number) {
  const el = document.getElementById(id) as HTMLDivElement | null;
  if (el) el.style.width = `${Math.min(Math.abs(value) / 5, 1) * 100}%`;
}

function drawChart(history: number[]) {
  const canvas = document.getElementById('reward-chart') as HTMLCanvasElement | null;
  const container = document.getElementById('reward-chart-container') as HTMLDivElement | null;
  if (!canvas || !container) return;
  canvas.width = container.clientWidth; canvas.height = container.clientHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (history.length === 0) return;
  const cum: number[] = []; let sum = 0;
  for (const r of history) { sum += r; cum.push(sum); }
  const maxT = Math.max(32, history.length);
  const maxR = Math.max(Math.abs(sum), 1);
  const px = 30, py = 10, pw = canvas.width - px - 10, ph = canvas.height - py * 2;
  ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.lineTo(px + pw, py + ph); ctx.stroke();
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < cum.length; i++) {
    const x = px + (i / maxT) * pw, y = py + ph - (cum[i] / maxR) * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#7c3aed';
  for (let i = 0; i < cum.length; i++) {
    const x = px + (i / maxT) * pw, y = py + ph - (cum[i] / maxR) * ph;
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  }
}
