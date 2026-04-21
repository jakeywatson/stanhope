import * as THREE from 'three';
import {
  type SceneController, type SceneObjects,
  createBaseRenderer, addBaseLighting, sleep, annotateEFE,
} from './types';

const CELL = 1.6;
const GRID = 15;
const CLASS_COLORS = {
  empty: new THREE.Color(0x181828),
  building: new THREE.Color(0x4b4b6e),
  decoy: new THREE.Color(0xef4444),
  target: new THREE.Color(0x22c55e),
};

export class DroneSceneV2 implements SceneController {
  private scene!: THREE.Scene;
  private drone!: THREE.Group;
  private glowLight!: THREE.PointLight;
  private targetPos = new THREE.Vector3();
  private driftTarget = new THREE.Vector3();

  private buildingMeshes: THREE.Mesh[] = [];
  private objectMarkers: Map<number, THREE.Mesh> = new Map();
  private objectLabels: Map<number, THREE.Sprite> = new Map();
  private objectLights: Map<number, THREE.PointLight> = new Map();

  private heatTiles: THREE.Mesh[][] = [];     // [x][y] grid of tiles
  private seenOverlay: Map<string, THREE.Mesh> = new Map();
  private frustumMesh: THREE.Mesh | null = null;
  private trailLine: THREE.Line | null = null;
  private trailPoints: THREE.Vector3[] = [];
  private outcomeOverlay: HTMLDivElement | null = null;

  private builtEnv = false;
  private scoreHistory: number[] = [];
  private missionEnded = false;
  private trainingScores: number[] = [];

  init(container: HTMLElement): SceneObjects {
    const { renderer, scene, camera, controls } = createBaseRenderer(container);
    this.scene = scene;

    const cx = (GRID * CELL) / 2;
    const cz = (GRID * CELL) / 2;
    camera.position.set(cx + 18, 18, cz + 18);
    camera.lookAt(cx, 2, cz);
    controls.target.set(cx, 1, cz);

    addBaseLighting(scene);
    scene.fog = new THREE.FogExp2(0x0a0a1a, 0.010);

    const groundSize = GRID * CELL + 12;
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(groundSize, groundSize),
      new THREE.MeshStandardMaterial({ color: 0x101020, roughness: 0.95 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -0.05, cz);
    ground.receiveShadow = true;
    scene.add(ground);

    this.buildHeatGrid();

    this.drone = this.buildDrone();
    const startX = 0 * CELL + CELL / 2;
    const startY = 0 * CELL + CELL / 2;
    const startZ = 2 * CELL + CELL / 2;
    this.drone.position.set(startX, startZ, startY);
    this.targetPos.copy(this.drone.position);
    scene.add(this.drone);

    this.glowLight = new THREE.PointLight(0x06b6d4, 3, 6);
    this.glowLight.position.copy(this.drone.position);
    scene.add(this.glowLight);

    const trailMat = new THREE.LineBasicMaterial({
      color: 0x06b6d4, transparent: true, opacity: 0.4,
    });
    this.trailLine = new THREE.Line(new THREE.BufferGeometry(), trailMat);
    scene.add(this.trailLine);

    return { scene, camera, renderer, controls, onFrame: () => this.onFrame() };
  }

  async animateStep(result: any): Promise<void> {
    const scene = result.scene;
    if (!this.builtEnv && scene) {
      this.buildEnvironment(scene);
      this.builtEnv = true;
    }

    this.updateObjects(scene, result);
    this.updateHeatmap(result.belief_summary);
    this.updateSeenCells(scene?.seen_cells ?? []);

    const [px, py, pz] = result.position;
    const target = new THREE.Vector3(
      px * CELL + CELL / 2,
      pz * CELL + CELL / 2,
      py * CELL + CELL / 2,
    );
    this.driftTarget.copy(target);
    await sleep(220);
    this.trailPoints.push(target.clone());
    this.updateTrail();
    this.updateFrustum(result.position, scene?.fov_radius ?? 1);

    const ended = result.declared_done === true || (result.battery ?? 999) <= 0;
    if (ended && !this.missionEnded) {
      this.missionEnded = true;
      this.showOutcomeOverlay(result);
    }
  }

  reset(): void {
    for (const m of this.buildingMeshes) this.scene.remove(m);
    this.buildingMeshes = [];
    for (const m of this.objectMarkers.values()) this.scene.remove(m);
    for (const m of this.objectLabels.values()) this.scene.remove(m);
    for (const m of this.objectLights.values()) this.scene.remove(m);
    this.objectMarkers.clear();
    this.objectLabels.clear();
    this.objectLights.clear();
    for (const m of this.seenOverlay.values()) this.scene.remove(m);
    this.seenOverlay.clear();
    if (this.frustumMesh) { this.scene.remove(this.frustumMesh); this.frustumMesh = null; }
    this.trailPoints = [];
    this.updateTrail();
    this.resetHeatGrid();
    this.builtEnv = false;
    this.missionEnded = false;
    this.scoreHistory = [];
    if (this.outcomeOverlay) { this.outcomeOverlay.remove(); this.outcomeOverlay = null; }
    const startX = 0 * CELL + CELL / 2;
    const startY = 0 * CELL + CELL / 2;
    const startZ = 2 * CELL + CELL / 2;
    this.drone.position.set(startX, startZ, startY);
    this.targetPos.copy(this.drone.position);
    this.driftTarget.copy(this.drone.position);
  }

  dispose(): void {}

  buildPanel(): string {
    return `
      <details class="theory-card">
        <summary>Paper Connection — Unknown-Site Search</summary>
        <div class="tc-body">
          <p>The drone arrives at a brand-new 15×15 site with <strong>no map</strong>. It must discover objects hidden behind buildings, discriminate real targets from decoys, and commit a confirm/reject decision before battery runs out.</p>
          <div class="tc-eq">G(π) = <span class="tc-ext">Extrinsic</span> (confirm right, reject wrong) + <span class="tc-sal">Salience</span> (resolve object identity) + <span class="tc-nov">Novelty</span> (learn site priors)</div>
          <p><span class="tc-label tc-ext">Extrinsic</span> = reward-weighted outcome: confirms target → +1, confirms decoy → −4 (harsh), declare-done prematurely → E[FN] penalty.</p>
          <p><span class="tc-label tc-sal">Salience</span> = expected entropy reduction on the 4-class cell belief and per-object target belief given a scan.</p>
          <p><span class="tc-label tc-nov">Novelty</span> = expected KL on the transferable Dirichlet α over cell classes — this is the term that lets the drone <strong>carry learning across sites</strong>.</p>
        </div>
      </details>

      <div class="efe-annotation" id="efe-annotation">
        <div class="ann-driver" id="ann-driver">Awaiting first step...</div>
        <div id="ann-detail"></div>
      </div>

      <div class="panel-section">
        <h3>Site Status</h3>
        <div style="font-size:0.72rem;color:#8080a0;line-height:1.7;">
          Discovered: <span id="v2-discovered" style="color:#06b6d4">0</span> / <span id="v2-n-objects">?</span>
          &nbsp;·&nbsp; Resolved: <span id="v2-resolved">0</span>
          &nbsp;·&nbsp; Seen cells: <span id="v2-seen">0</span>
        </div>
        <div style="font-size:0.72rem;color:#8080a0;line-height:1.7;">
          TP: <span id="v2-tp" style="color:#22c55e">0</span>
          &nbsp;·&nbsp; FP: <span id="v2-fp" style="color:#ef4444">0</span>
          &nbsp;·&nbsp; TN: <span id="v2-tn" style="color:#c0c0e0">0</span>
          &nbsp;·&nbsp; FN: <span id="v2-fn" style="color:#f59e0b">0</span>
        </div>
        <div style="font-size:0.72rem;color:#8080a0;line-height:1.7;">
          Score so far: <span id="v2-score" style="color:#06b6d4">0.0</span>
        </div>
      </div>

      <div class="panel-section">
        <h3>Object Beliefs — P(target)</h3>
        <div id="v2-obj-beliefs"></div>
      </div>

      <div class="panel-section">
        <h3>Learned Cell-Class Prior (α)</h3>
        <div id="v2-alpha"></div>
        <div style="display:flex;gap:0.4rem;margin-top:0.5rem;">
          <button id="v2-btn-train" style="flex:1;background:#1a1a35;color:#d8d8ee;border:1px solid #2a2a4a;padding:0.35rem 0.5rem;border-radius:4px;font-size:0.7rem;cursor:pointer;font-family:inherit;">Train 10 eps</button>
          <button id="v2-btn-reset-alpha" style="flex:1;background:#1a1a35;color:#d8d8ee;border:1px solid #2a2a4a;padding:0.35rem 0.5rem;border-radius:4px;font-size:0.7rem;cursor:pointer;font-family:inherit;">Reset α</button>
        </div>
        <div id="v2-train-status" style="font-size:0.65rem;color:#606080;margin-top:0.4rem;line-height:1.5;">
          Carries across episodes. Drives the novelty term — observing new sites reduces this prior's entropy.
        </div>
        <div id="v2-train-curve-container" style="height:70px;margin-top:0.5rem;"><canvas id="v2-train-curve"></canvas></div>
      </div>

      <div class="panel-section">
        <h3>Drone Status</h3>
        <div class="belief-bar"><label>Altitude</label><div class="bar-track"><div class="bar-fill" id="v2-bar-alt" style="width:50%;background:#a78bfa"></div></div><span class="bar-value" id="v2-val-alt">2</span></div>
        <div class="belief-bar"><label>Battery</label><div class="bar-track"><div class="bar-fill" id="v2-bar-batt" style="width:100%;background:#22c55e"></div></div><span class="bar-value" id="v2-val-batt">100%</span></div>
        <div style="font-size:0.72rem;color:#8080a0;margin-top:0.3rem;">
          FOV radius: <span id="v2-fov">1</span> cells &nbsp;·&nbsp; Step: <span id="v2-step">0</span> &nbsp;·&nbsp; Episode: <span id="v2-ep" style="color:#06b6d4">0</span>
        </div>
      </div>

      <div class="panel-section">
        <h3>Current Waypoint</h3>
        <div id="v2-wp" style="font-size:1.0rem;font-weight:700;color:#06b6d4;margin-bottom:0.3rem;">—</div>
        <div class="efe-row"><label>Extrinsic</label><div class="efe-bar"><div class="efe-fill fill-extrinsic" id="v2-efe-ext" style="width:0%"></div></div></div>
        <div class="efe-row"><label>Salience</label><div class="efe-bar"><div class="efe-fill fill-salience" id="v2-efe-sal" style="width:0%"></div></div></div>
        <div class="efe-row"><label>Novelty</label><div class="efe-bar"><div class="efe-fill fill-novelty" id="v2-efe-nov" style="width:0%"></div></div></div>
      </div>

      <div class="panel-section">
        <h3>Waypoint Policy (softmax)</h3>
        <div id="v2-wp-policy" style="max-height:160px;overflow-y:auto;"></div>
      </div>

      <div class="panel-section">
        <h3>Mission Score</h3>
        <div id="v2-score-chart-container"><canvas id="v2-score-chart"></canvas></div>
      </div>
    `;
  }

  updatePanel(result: any): void {
    const scene = result.scene;
    const objects = scene?.objects ?? [];
    const alpha = result.world_alpha ?? [];
    const classNames = result.class_names ?? ['empty', 'building', 'decoy', 'target'];

    // Site status
    setText('v2-discovered', String(result.discovered ?? 0));
    setText('v2-n-objects', String(result.n_objects ?? objects.length));
    setText('v2-resolved', String(result.n_resolved ?? 0));
    setText('v2-seen', String(scene?.seen_cells?.length ?? 0));
    const oc = result.outcome_counts ?? {};
    setText('v2-tp', String(oc.tp ?? 0));
    setText('v2-fp', String(oc.fp ?? 0));
    setText('v2-tn', String(oc.tn ?? 0));
    setText('v2-fn', String(oc.fn ?? 0));
    setText('v2-score', (result.total_reward ?? 0).toFixed(2));

    // Per-object beliefs
    const objDiv = document.getElementById('v2-obj-beliefs');
    if (objDiv) {
      if (!objects.length) {
        objDiv.innerHTML = '<div style="font-size:0.68rem;color:#606080">No objects discovered yet.</div>';
      } else {
        objDiv.innerHTML = objects.map((o: any) => {
          const p = o.target_belief ?? 0.5;
          const discovered = o.discovered;
          const resolved = o.resolved;
          const tagCol = discovered
            ? (p > 0.7 ? '#22c55e' : p < 0.3 ? '#ef4444' : '#f59e0b')
            : '#404060';
          let status = '';
          if (resolved === 'confirmed') status = ' <span style="color:#22c55e">✓ confirmed</span>';
          else if (resolved === 'rejected') status = ' <span style="color:#ef4444">✗ rejected</span>';
          else if (discovered) status = ' <span style="color:#a78bfa">discovered</span>';
          else status = ' <span style="color:#404060">unseen</span>';
          return `<div class="belief-bar">
            <label style="color:${tagCol}">obj ${o.idx} (${o.x},${o.y})${status}</label>
            <div class="bar-track"><div class="bar-fill" style="width:${p * 100}%;background:${tagCol}"></div></div>
            <span class="bar-value">${p.toFixed(2)}</span>
          </div>`;
        }).join('');
      }
    }

    // Dirichlet α bars
    const alphaDiv = document.getElementById('v2-alpha');
    if (alphaDiv && alpha.length) {
      const sum = alpha.reduce((a: number, b: number) => a + b, 0) || 1;
      const colors = ['#6b7280', '#a78bfa', '#ef4444', '#22c55e'];
      alphaDiv.innerHTML = alpha.map((a: number, i: number) => {
        const pct = (a / sum) * 100;
        return `<div class="belief-bar">
          <label style="color:${colors[i]}">${classNames[i]}</label>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${colors[i]}"></div></div>
          <span class="bar-value">${a.toFixed(0)}</span>
        </div>`;
      }).join('');
    }

    // Drone status
    const alt = result.position?.[2] ?? 2;
    setBar('v2-bar-alt', 'v2-val-alt', (alt / 4) * 100, String(alt));
    const battMax = scene?.battery_max ?? 200;
    const battPct = Math.max(0, Math.min(100, ((result.battery ?? battMax) / battMax) * 100));
    const battColor = battPct > 40 ? '#22c55e' : battPct > 15 ? '#f59e0b' : '#ef4444';
    const battBar = document.getElementById('v2-bar-batt') as HTMLDivElement | null;
    const battVal = document.getElementById('v2-val-batt') as HTMLSpanElement | null;
    if (battBar) { battBar.style.width = `${battPct}%`; battBar.style.background = battColor; }
    if (battVal) battVal.textContent = `${Math.round(battPct)}%`;
    setText('v2-fov', String(scene?.fov_radius ?? '?'));
    setText('v2-step', String(result.step ?? 0));
    setText('v2-ep', String(scene?.episode ?? 0));

    // Waypoint + EFE
    setText('v2-wp', result.waypoint ?? '—');
    const wpEl = document.getElementById('v2-wp');
    if (wpEl) wpEl.style.color = wpColor(result.waypoint ?? '');
    const chosenEfe = result.efe?.[result.waypoint];
    if (chosenEfe) {
      setEFE('v2-efe-ext', chosenEfe.extrinsic);
      setEFE('v2-efe-sal', chosenEfe.salience);
      setEFE('v2-efe-nov', chosenEfe.novelty);
    } else {
      setEFE('v2-efe-ext', 0); setEFE('v2-efe-sal', 0); setEFE('v2-efe-nov', 0);
    }

    // Policy probs
    const polDiv = document.getElementById('v2-wp-policy');
    if (polDiv) {
      const entries = Object.entries(result.policy_probs ?? {})
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 10);
      polDiv.innerHTML = entries.map(([name, prob]) => {
        const p = prob as number;
        const c = wpColor(name);
        return `<div class="belief-bar"><label style="color:${c};font-size:0.65rem">${name}</label><div class="bar-track"><div class="bar-fill" style="width:${p * 100}%;background:${c}"></div></div><span class="bar-value">${p.toFixed(2)}</span></div>`;
      }).join('');
    }

    // Score history
    this.scoreHistory.push(result.total_reward ?? 0);
    drawScoreChart(this.scoreHistory);

    annotateEFE(result.efe, result.waypoint, {
      extrinsic: 'The drone commits — confirming a target or rejecting a decoy.',
      salience: 'The drone scans or explores to sharpen in-episode class/identity beliefs.',
      novelty: 'The drone gathers observations that improve its transferable site prior α.',
      tie: 'Multiple objectives align — the drone balances commitment with curiosity.',
    });
  }

  resetPanel(): void {
    this.scoreHistory = [];
    for (const id of ['v2-obj-beliefs', 'v2-alpha', 'v2-wp-policy']) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = '';
    }
    for (const [id, label] of [['v2-bar-alt', '2'], ['v2-bar-batt', '100%']] as const) {
      const bar = document.getElementById(id) as HTMLDivElement | null;
      if (bar) bar.style.width = id === 'v2-bar-batt' ? '100%' : '50%';
    }
    setText('v2-val-alt', '2'); setText('v2-val-batt', '100%');
    setEFE('v2-efe-ext', 0); setEFE('v2-efe-sal', 0); setEFE('v2-efe-nov', 0);
    setText('v2-wp', '—');
    drawScoreChart([]);
    this.drawTrainingCurve();
  }

  /** External entry points for the training workflow. */
  appendTrainingScores(scores: number[]): void {
    this.trainingScores.push(...scores);
    this.drawTrainingCurve();
  }

  clearTrainingCurve(): void {
    this.trainingScores = [];
    this.drawTrainingCurve();
  }

  updateAlphaDisplay(alpha: number[], classNames: string[] = ['empty', 'building', 'decoy', 'target']): void {
    const alphaDiv = document.getElementById('v2-alpha');
    if (!alphaDiv || !alpha.length) return;
    const sum = alpha.reduce((a, b) => a + b, 0) || 1;
    const colors = ['#6b7280', '#a78bfa', '#ef4444', '#22c55e'];
    alphaDiv.innerHTML = alpha.map((a, i) => {
      const pct = (a / sum) * 100;
      return `<div class="belief-bar">
        <label style="color:${colors[i]}">${classNames[i]}</label>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${colors[i]}"></div></div>
        <span class="bar-value">${a.toFixed(0)}</span>
      </div>`;
    }).join('');
  }

  setTrainStatus(msg: string): void {
    const el = document.getElementById('v2-train-status');
    if (el) el.textContent = msg;
  }

  private drawTrainingCurve(): void {
    const canvas = document.getElementById('v2-train-curve') as HTMLCanvasElement | null;
    const container = document.getElementById('v2-train-curve-container') as HTMLDivElement | null;
    if (!canvas || !container) return;
    canvas.width = container.clientWidth; canvas.height = container.clientHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const h = this.trainingScores;
    if (!h.length) {
      ctx.fillStyle = '#505070';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText('No training episodes yet', 6, 14);
      return;
    }
    const pad = 6;
    const pw = canvas.width - pad * 2;
    const ph = canvas.height - pad * 2;
    const maxAbs = Math.max(2, ...h.map(Math.abs));
    ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
    const zeroY = pad + ph / 2;
    ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(pad + pw, zeroY); ctx.stroke();
    // running mean
    const window = 5;
    const smoothed = h.map((_, i) => {
      const lo = Math.max(0, i - window + 1);
      const slice = h.slice(lo, i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });
    ctx.strokeStyle = '#06b6d4'; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < smoothed.length; i++) {
      const x = pad + (i / Math.max(h.length - 1, 1)) * pw;
      const y = zeroY - (smoothed[i] / maxAbs) * (ph / 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    // raw dots
    ctx.fillStyle = '#a78bfa88';
    for (let i = 0; i < h.length; i++) {
      const x = pad + (i / Math.max(h.length - 1, 1)) * pw;
      const y = zeroY - (h[i] / maxAbs) * (ph / 2);
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
  }

  // ─── Scene building ───

  private buildHeatGrid() {
    this.heatTiles = [];
    for (let x = 0; x < GRID; x++) {
      this.heatTiles[x] = [];
      for (let y = 0; y < GRID; y++) {
        const mat = new THREE.MeshBasicMaterial({
          color: 0x0a0a18, transparent: true, opacity: 0.0, depthWrite: false,
        });
        const geo = new THREE.PlaneGeometry(CELL * 0.96, CELL * 0.96);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(x * CELL + CELL / 2, 0.02, y * CELL + CELL / 2);
        this.scene.add(mesh);
        this.heatTiles[x][y] = mesh;
      }
    }
  }

  private resetHeatGrid() {
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        const t = this.heatTiles[x]?.[y];
        if (!t) continue;
        const mat = t.material as THREE.MeshBasicMaterial;
        mat.color.setHex(0x0a0a18);
        mat.opacity = 0.0;
      }
    }
  }

  private buildEnvironment(scene: any) {
    const bldgMat = new THREE.MeshStandardMaterial({
      color: 0x2b2b44, roughness: 0.65, metalness: 0.18,
    });
    for (const b of (scene.buildings ?? [])) {
      const h = b.h * CELL;
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(CELL * 0.82, h, CELL * 0.82), bldgMat);
      mesh.position.set(b.x * CELL + CELL / 2, h / 2, b.y * CELL + CELL / 2);
      mesh.castShadow = true; mesh.receiveShadow = true;
      this.scene.add(mesh);
      this.buildingMeshes.push(mesh);
    }
  }

  private updateObjects(scene: any, result: any) {
    const objects = scene?.objects ?? [];
    const ended = result.declared_done === true || (result.battery ?? 999) <= 0;
    for (const o of objects) {
      const show = o.discovered || ended;
      if (!show) {
        this.removeObject(o.idx);
        continue;
      }
      this.upsertObject(o, ended);
    }
  }

  private upsertObject(o: any, ended: boolean) {
    const beliefCol = new THREE.Color().lerpColors(
      new THREE.Color(0xef4444),  // decoy (low p)
      new THREE.Color(0x22c55e),  // target (high p)
      Math.max(0, Math.min(1, o.target_belief)),
    );
    const resolveCol = o.resolved === 'confirmed' ? 0x22c55e
                     : o.resolved === 'rejected'  ? 0x6b7280
                     : null;
    const trueCol = o.is_target ? 0x22c55e : 0xef4444;
    const finalCol = resolveCol !== null ? resolveCol : beliefCol.getHex();

    let marker = this.objectMarkers.get(o.idx);
    if (!marker) {
      const mat = new THREE.MeshStandardMaterial({
        color: finalCol, emissive: finalCol, emissiveIntensity: 0.55,
        roughness: 0.25, metalness: 0.3,
      });
      marker = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.3, 0.8, 12), mat);
      marker.position.set(o.x * CELL + CELL / 2, 0.4, o.y * CELL + CELL / 2);
      marker.castShadow = true;
      this.scene.add(marker);
      this.objectMarkers.set(o.idx, marker);

      const light = new THREE.PointLight(finalCol, 1.0, 3.2);
      light.position.set(o.x * CELL + CELL / 2, 0.9, o.y * CELL + CELL / 2);
      this.scene.add(light);
      this.objectLights.set(o.idx, light);
    }
    const mat = marker.material as THREE.MeshStandardMaterial;
    mat.color.setHex(finalCol);
    mat.emissive.setHex(finalCol);
    mat.emissiveIntensity = 0.3 + Math.abs(o.target_belief - 0.5) * 1.2;
    const light = this.objectLights.get(o.idx);
    if (light) light.color.setHex(finalCol);

    // Label
    let label = this.objectLabels.get(o.idx);
    const labelText = ended
      ? `obj${o.idx} ${o.is_target ? '★' : '·'}${o.resolved === 'confirmed' ? ' ✓' : o.resolved === 'rejected' ? ' ✗' : ''}`
      : `obj${o.idx}`;
    const labelColor = ended
      ? (o.is_target ? '#22c55e' : '#ef4444')
      : '#d8d8ee';
    if (label) { this.scene.remove(label); this.objectLabels.delete(o.idx); }
    label = makeTextSprite(labelText, labelColor);
    label.position.set(o.x * CELL + CELL / 2, 1.35, o.y * CELL + CELL / 2);
    this.scene.add(label);
    this.objectLabels.set(o.idx, label);

    if (ended && o.is_target) {
      mat.emissive.setHex(trueCol);
      mat.emissiveIntensity = 1.0;
    }
  }

  private removeObject(idx: number) {
    const m = this.objectMarkers.get(idx);
    if (m) { this.scene.remove(m); this.objectMarkers.delete(idx); }
    const l = this.objectLabels.get(idx);
    if (l) { this.scene.remove(l); this.objectLabels.delete(idx); }
    const light = this.objectLights.get(idx);
    if (light) { this.scene.remove(light); this.objectLights.delete(idx); }
  }

  private updateHeatmap(summary: any) {
    if (!summary?.cells) return;
    for (const c of summary.cells) {
      const tile = this.heatTiles[c.x]?.[c.y];
      if (!tile) continue;
      const mat = tile.material as THREE.MeshBasicMaterial;
      const p = c.p as [number, number, number, number];
      // Blend class colors weighted by belief.
      const col = new THREE.Color(
        CLASS_COLORS.empty.r * p[0] + CLASS_COLORS.building.r * p[1]
          + CLASS_COLORS.decoy.r * p[2] + CLASS_COLORS.target.r * p[3],
        CLASS_COLORS.empty.g * p[0] + CLASS_COLORS.building.g * p[1]
          + CLASS_COLORS.decoy.g * p[2] + CLASS_COLORS.target.g * p[3],
        CLASS_COLORS.empty.b * p[0] + CLASS_COLORS.building.b * p[1]
          + CLASS_COLORS.decoy.b * p[2] + CLASS_COLORS.target.b * p[3],
      );
      // Entropy → opacity: low entropy (confident) = visible, high entropy = faint.
      let H = 0;
      for (const pi of p) { if (pi > 1e-6) H -= pi * Math.log(pi); }
      const Hmax = Math.log(4);
      const confidence = 1 - H / Hmax;
      mat.color.copy(col);
      mat.opacity = 0.15 + confidence * 0.55;
    }
  }

  private updateSeenCells(cells: number[][]) {
    for (const [x, y] of cells) {
      const key = `${x},${y}`;
      if (this.seenOverlay.has(key)) continue;
      const mat = new THREE.MeshBasicMaterial({
        color: 0x06b6d4, transparent: true, opacity: 0.06, depthWrite: false,
      });
      const geo = new THREE.PlaneGeometry(CELL * 0.92, CELL * 0.92);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(x * CELL + CELL / 2, 0.05, y * CELL + CELL / 2);
      this.scene.add(mesh);
      this.seenOverlay.set(key, mesh);
    }
  }

  private updateFrustum(pos: number[], radius: number) {
    if (this.frustumMesh) this.scene.remove(this.frustumMesh);
    const [px, py, pz] = pos;
    const r = Math.max(0.2, radius) * CELL;
    const h = pz * CELL;
    const mat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4, transparent: true, opacity: 0.07, side: THREE.DoubleSide,
    });
    const geo = new THREE.ConeGeometry(r, h, 20, 1, true);
    this.frustumMesh = new THREE.Mesh(geo, mat);
    this.frustumMesh.position.set(px * CELL + CELL / 2, h / 2, py * CELL + CELL / 2);
    this.scene.add(this.frustumMesh);
  }

  private updateTrail() {
    if (!this.trailLine || this.trailPoints.length < 2) return;
    const positions = new Float32Array(this.trailPoints.length * 3);
    for (let i = 0; i < this.trailPoints.length; i++) {
      positions[i * 3] = this.trailPoints[i].x;
      positions[i * 3 + 1] = this.trailPoints[i].y;
      positions[i * 3 + 2] = this.trailPoints[i].z;
    }
    this.trailLine.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.trailLine.geometry = geo;
  }

  private buildDrone(): THREE.Group {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x06b6d4, metalness: 0.7, roughness: 0.2,
      emissive: 0x06b6d4, emissiveIntensity: 0.4,
    });
    g.add(new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 0.14, 8), bodyMat));
    const armMat = new THREE.MeshStandardMaterial({ color: 0x404060, metalness: 0.5, roughness: 0.3 });
    const rotorMat = new THREE.MeshStandardMaterial({ color: 0x808090, transparent: true, opacity: 0.7 });
    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
      const dist = 0.40;
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.04, dist * 2), armMat);
      arm.rotation.y = angle;
      arm.position.set(Math.cos(angle) * dist * 0.5, 0, Math.sin(angle) * dist * 0.5);
      g.add(arm);
      const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.02, 12), rotorMat);
      rotor.position.set(Math.cos(angle) * dist, 0.1, Math.sin(angle) * dist);
      rotor.userData.rotorSpeed = 0.3 + Math.random() * 0.1;
      g.add(rotor);
    }
    return g;
  }

  private onFrame() {
    const delta = this.driftTarget.clone().sub(this.drone.position);
    if (delta.length() > 0.03) this.drone.position.add(delta.multiplyScalar(0.12));
    this.drone.position.y += Math.sin(Date.now() * 0.005) * 0.003;
    this.glowLight.position.copy(this.drone.position);
    for (const child of this.drone.children) {
      if (child.userData.rotorSpeed) child.rotation.y += child.userData.rotorSpeed;
    }
    for (const m of this.objectMarkers.values()) {
      m.position.y = 0.4 + Math.sin(Date.now() * 0.003) * 0.03;
    }
  }

  private showOutcomeOverlay(result: any): void {
    const viewport = document.getElementById('viewport');
    if (!viewport) return;
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;pointer-events:none;z-index:10;';

    const oc = result.outcome_counts ?? {};
    const score = result.terminal_score ?? result.total_reward ?? 0;
    const success = (oc.fp ?? 0) === 0 && (oc.fn ?? 0) === 0 && (oc.tp ?? 0) > 0;
    const batteryOut = (result.battery ?? 999) <= 0 && !result.declared_done;
    const icon = batteryOut ? '⚡' : success ? '✓' : (oc.fp ?? 0) > 0 ? '✗' : '—';
    const color = batteryOut ? '#f59e0b' : success ? '#22c55e' : (oc.fp ?? 0) > 0 ? '#ef4444' : '#a78bfa';
    const bg = `${color}1a`;
    const title = batteryOut ? 'BATTERY DEPLETED'
                : success ? 'SITE CLEARED'
                : (oc.fp ?? 0) > 0 ? 'WRONG CONFIRMATION'
                : 'SITE DECLARED DONE';
    const sub = `TP ${oc.tp ?? 0} · FP ${oc.fp ?? 0} · TN ${oc.tn ?? 0} · FN ${oc.fn ?? 0}  ·  score ${score.toFixed(2)}  ·  ${result.step ?? 0} steps`;

    overlay.innerHTML = `
      <div style="background:${bg};backdrop-filter:blur(4px);border:2px solid ${color};border-radius:12px;padding:1.4rem 2.2rem;text-align:center;animation:fadeInScale 0.4s ease-out">
        <div style="font-size:2.5rem;color:${color};margin-bottom:0.3rem">${icon}</div>
        <div style="font-size:1.05rem;font-weight:700;color:${color};letter-spacing:0.05em">${title}</div>
        <div style="font-size:0.8rem;color:#a0a0c0;margin-top:0.35rem">${sub}</div>
      </div>
    `;
    viewport.style.position = 'relative';
    viewport.appendChild(overlay);
    this.outcomeOverlay = overlay;
  }
}

// ─── Helpers ───

function setText(id: string, text: string) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setBar(barId: string, valId: string, pct: number, label: string) {
  const bar = document.getElementById(barId) as HTMLDivElement | null;
  const val = document.getElementById(valId) as HTMLSpanElement | null;
  if (bar) bar.style.width = `${Math.min(pct, 100)}%`;
  if (val) val.textContent = label;
}

function setEFE(id: string, value: number) {
  const el = document.getElementById(id) as HTMLDivElement | null;
  if (el) el.style.width = `${Math.min(Math.abs(value) / 5, 1) * 100}%`;
}

function wpColor(name: string): string {
  if (!name) return '#8080a0';
  if (name.startsWith('Explore')) return '#06b6d4';
  if (name.startsWith('Scan')) return '#3b82f6';
  if (name.startsWith('Confirm')) return '#22c55e';
  if (name.startsWith('Reject')) return '#f59e0b';
  if (name.startsWith('Declare')) return '#a78bfa';
  return '#d8d8ee';
}

function makeTextSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.font = 'bold 32px sans-serif';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);
  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.6, 0.4, 1);
  return sprite;
}

function drawScoreChart(history: number[]) {
  const canvas = document.getElementById('v2-score-chart') as HTMLCanvasElement | null;
  const container = document.getElementById('v2-score-chart-container') as HTMLDivElement | null;
  if (!canvas || !container) return;
  canvas.width = container.clientWidth; canvas.height = container.clientHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (history.length === 0) return;
  const maxAbs = Math.max(4, ...history.map(Math.abs));
  const px = 30, py = 10, pw = canvas.width - px - 10, ph = canvas.height - py * 2;
  ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
  // zero line
  ctx.beginPath();
  const zeroY = py + ph / 2;
  ctx.moveTo(px, zeroY); ctx.lineTo(px + pw, zeroY); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.stroke();

  ctx.strokeStyle = '#06b6d4'; ctx.lineWidth = 2; ctx.beginPath();
  const maxT = Math.max(50, history.length);
  for (let i = 0; i < history.length; i++) {
    const x = px + (i / maxT) * pw;
    const y = zeroY - (history[i] / maxAbs) * (ph / 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}
