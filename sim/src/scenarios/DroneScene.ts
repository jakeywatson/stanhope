import * as THREE from 'three';
import {
  type SceneController, type SceneObjects,
  createBaseRenderer, addBaseLighting, sleep, annotateEFE,
} from './types';

const CELL = 2.0;
const ACTION_NAMES = ['north', 'south', 'east', 'west', 'up', 'down', 'confirm'];
const ACT_COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#a78bfa', '#f472b6', '#facc15'];
const OBJ_COLORS = [0x22c55e, 0x3b82f6, 0xf59e0b, 0xef4444, 0xa78bfa, 0xf472b6];
const OBJ_CSS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#a78bfa', '#f472b6'];

export class DroneScene implements SceneController {
  private scene!: THREE.Scene;
  private drone!: THREE.Group;
  private glowLight!: THREE.PointLight;
  private frustumMesh: THREE.Mesh | null = null;
  private buildingMeshes: THREE.Mesh[] = [];
  private objectMarkers: THREE.Mesh[] = [];
  private objectLabels: THREE.Sprite[] = [];
  private objectLights: THREE.PointLight[] = [];
  private trailPoints: THREE.Vector3[] = [];
  private trailLine: THREE.Line | null = null;
  private targetPos = new THREE.Vector3(4 * CELL, 3 * CELL, 4 * CELL);
  private gridW = 12; private gridD = 12;
  private rewardHistory: number[] = [];
  private builtEnv = false;
  private outcomeOverlay: HTMLDivElement | null = null;
  private missionEnded = false;
  private createdObjects = new Set<number>();
  private exploredTiles: Map<string, THREE.Mesh> = new Map();
  private frontierTiles: THREE.Mesh[] = [];
  private altitudePole: THREE.Line | null = null;
  private altitudeLabel: THREE.Sprite | null = null;
  private efeArrow: THREE.ArrowHelper | null = null;
  private observationFlash: THREE.Mesh | null = null;
  private observationFlashUntil = 0;

  init(container: HTMLElement): SceneObjects {
    const { renderer, scene, camera, controls } = createBaseRenderer(container);
    this.scene = scene;

    const cx = (this.gridW * CELL) / 2;
    const cz = (this.gridD * CELL) / 2;
    camera.position.set(cx + 18, 16, cz + 18);
    camera.lookAt(cx, 3, cz);
    controls.target.set(cx, 3, cz);

    addBaseLighting(scene);
    scene.fog = new THREE.FogExp2(0x0a0a1a, 0.012);

    const gSize = Math.max(this.gridW, this.gridD) * CELL + 16;
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(gSize, gSize),
      new THREE.MeshStandardMaterial({ color: 0x181828, roughness: 0.95 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -0.05, cz);
    ground.receiveShadow = true;
    scene.add(ground);

    this.drone = this.buildDrone();
    this.drone.position.copy(this.targetPos);
    scene.add(this.drone);

    this.glowLight = new THREE.PointLight(0x06b6d4, 3, 6);
    this.glowLight.position.copy(this.targetPos);
    scene.add(this.glowLight);

    const trailGeo = new THREE.BufferGeometry();
    const trailMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.35 });
    this.trailLine = new THREE.Line(trailGeo, trailMat);
    scene.add(this.trailLine);

    return { scene, camera, renderer, controls, onFrame: () => this.onFrame() };
  }

  async animateStep(result: any): Promise<void> {
    const ss = result.scan_state;
    if (!this.builtEnv && ss) {
      this.buildEnvironment(ss);
      this.builtEnv = true;
    }

    // Update object marker colors based on belief
    this.updateObjectBeliefs(result);

    // Move drone: Python [x, y, z] → Three.js X=x, Y=z(alt), Z=y
    const [px, py, pz] = result.position;
    const target = new THREE.Vector3(px * CELL + CELL / 2, pz * CELL + CELL / 2, py * CELL + CELL / 2);
    this.targetPos.copy(target);
    await sleep(300);
    this.trailPoints.push(target.clone());
    this.updateTrail();

    this.updateFrustum(result.position, ss?.fov_radius ?? 2);
    this.updateExploredTiles(ss?.seen_cells ?? []);
    this.updateFrontierHalo(ss, result);
    this.updateAltitudePole(result.position);
    this.updateEfeArrow(result);
    this.flashObservation(result);

    // Show outcome overlay on mission end
    if (!this.missionEnded && (result.found_target || result.mission_failed || (result.battery ?? 999) <= 0)) {
      this.missionEnded = true;
      this.showOutcomeOverlay(result);
    }
  }

  reset(): void {
    for (const m of this.buildingMeshes) this.scene.remove(m);
    this.buildingMeshes = [];
    for (const m of this.objectMarkers) this.scene.remove(m);
    this.objectMarkers = [];
    for (const s of this.objectLabels) this.scene.remove(s);
    this.objectLabels = [];
    for (const l of this.objectLights) this.scene.remove(l);
    this.objectLights = [];
    if (this.frustumMesh) { this.scene.remove(this.frustumMesh); this.frustumMesh = null; }
    for (const t of this.exploredTiles.values()) this.scene.remove(t);
    this.exploredTiles.clear();
    for (const t of this.frontierTiles) this.scene.remove(t);
    this.frontierTiles = [];
    if (this.altitudePole) { this.scene.remove(this.altitudePole); this.altitudePole = null; }
    if (this.altitudeLabel) { this.scene.remove(this.altitudeLabel); this.altitudeLabel = null; }
    if (this.efeArrow) { this.scene.remove(this.efeArrow); this.efeArrow = null; }
    if (this.observationFlash) { this.scene.remove(this.observationFlash); this.observationFlash = null; }
    this.trailPoints = [];
    this.updateTrail();
    this.builtEnv = false;
    this.missionEnded = false;
    this.createdObjects.clear();
    if (this.outcomeOverlay) { this.outcomeOverlay.remove(); this.outcomeOverlay = null; }
    this.drone.position.set(4 * CELL, 3 * CELL, 4 * CELL);
    this.targetPos.copy(this.drone.position);
    this.rewardHistory = [];
  }

  dispose(): void {}

  // ─── Panel ───

  buildPanel(): string {
    return `
      <details class="theory-card">
        <summary>Paper Connection — Search & Discriminate</summary>
        <div class="tc-body">
          <p>A drone must <strong>find and identify a target object</strong> among 6 similar distractors in a 12×12 grid — directly modelling <a href="https://stanhopeai.com" style="color:#06b6d4">Stanhope AI's</a> FEM world model for autonomous drone perception.</p>
          <div class="tc-eq">G(π) = <span class="tc-ext">Extrinsic</span> (confirm target) + <span class="tc-sal">Salience</span> (explore & inspect) + <span class="tc-nov">Novelty</span> (learn sensor model)</div>
          <p><strong>Phase 1 — Search:</strong> The drone knows 6 objects exist but not <em>where</em>. It generates <strong>Explore</strong> waypoints from the unseen frontier and scores them using learned priors about which altitudes discover objects fastest, how strongly buildings block the view, and which <strong>relative search moves</strong> tend to work best.</p>
          <p><strong>Phase 2 — Discriminate:</strong> Once objects are discovered, <strong>Scan</strong> waypoints navigate to each at z=2 — the sweet spot for sensor discrimination. The drone weighs info gain vs. distance cost.</p>
          <p><strong>Phase 3 — Confirm:</strong> When belief is high enough, the <strong>Confirm</strong> waypoint navigates to the object at z=1 and triggers confirmation. A wrong confirmation ends the mission.</p>
          <p><span class="tc-label tc-sal">Salience</span> drives both <strong>exploration</strong> (scanning unseen areas) and <strong>discrimination</strong> (inspecting objects at z=2).</p>
          <p><span class="tc-label tc-nov">Novelty</span> drives altitude exploration — the drone learns which altitudes are most revealing.</p>
          <p><span class="tc-label tc-ext">Extrinsic</span> drives confirmation — once confident, descend and confirm.</p>
        </div>
      </details>
      <div class="efe-annotation" id="efe-annotation">
        <div class="ann-driver" id="ann-driver">Awaiting first step...</div>
        <div id="ann-detail"></div>
      </div>
      <div class="panel-section">
        <h3>Object Beliefs — P(target = object)</h3>
        <div id="obj-beliefs"></div>
        <div style="font-size:0.68rem;color:#606080;margin-top:0.4rem;">
          Discovered: <span id="discovered-count" style="color:#06b6d4">0</span> / <span id="total-objects">6</span>
          &nbsp;·&nbsp; Inspected: <span id="inspected-count">0</span>
          &nbsp;·&nbsp; Explored: <span id="explored-pct">0</span>%
        </div>
      </div>
      <div class="panel-section">
        <h3>Observation Log</h3>
        <div id="obs-log" style="font-size:0.68rem;color:#8080a0;max-height:120px;overflow-y:auto;line-height:1.6;"></div>
      </div>
      <div class="panel-section">
        <h3>Drone Status</h3>
        <div class="belief-bar"><label>Altitude</label><div class="bar-track"><div class="bar-fill" id="bar-alt" style="width:75%;background:#a78bfa"></div></div><span class="bar-value" id="val-alt">3</span></div>
        <div class="belief-bar"><label>Battery</label><div class="bar-track"><div class="bar-fill" id="bar-batt" style="width:100%;background:#22c55e"></div></div><span class="bar-value" id="val-batt">100%</span></div>
        <div style="font-size:0.75rem;color:#8080a0;margin-top:0.3rem;">
          FOV radius: <span id="fov-radius">3</span> cells | Step: <span id="drone-step">0</span> | Episode: <span id="drone-episode" style="color:#06b6d4">0</span>
        </div>
        <div style="font-size:0.70rem;color:#8080a0;margin-top:0.3rem;">
          Target: <span id="target-status" style="color:#ef4444">searching...</span>
        </div>
      </div>
      <div class="panel-section">
        <h3>Sensor Model (learned)</h3>
        ${[1, 2, 3, 4].map(z => `
          <div class="belief-bar"><label style="font-size:0.7rem">z=${z}</label><div class="bar-track"><div class="bar-fill fill-novelty" id="disc-z${z}" style="width:60%"></div></div><span class="bar-value" id="disc-val-z${z}">0.60</span></div>
        `).join('')}
      </div>
      <details class="panel-section adv-details">
        <summary class="adv-summary">Search Model (learned) — advanced</summary>
        <div style="margin-top:0.5rem">
          ${[2, 3, 4].map(z => `
            <div class="belief-bar"><label style="font-size:0.78rem">z=${z} gain</label><div class="bar-track"><div class="bar-fill fill-salience" id="search-z${z}" style="width:40%"></div></div><span class="bar-value" id="search-val-z${z}">1.00</span></div>
            <div class="belief-bar"><label style="font-size:0.78rem">z=${z} clear</label><div class="bar-track"><div class="bar-fill fill-extrinsic" id="clear-z${z}" style="width:80%"></div></div><span class="bar-value" id="clear-val-z${z}">0.80</span></div>
          `).join('')}
        </div>
      </details>
      <details class="panel-section adv-details">
        <summary class="adv-summary">Direction Model (learned) — advanced</summary>
        <div id="dir-profile" style="margin-top:0.5rem"></div>
      </details>
      <div class="panel-section">
        <h3>Current Waypoint</h3>
        <div id="current-wp" style="font-size:1.0rem;font-weight:700;color:#06b6d4;margin-bottom:0.3rem;">—</div>
        <div style="font-size:0.70rem;color:#8080a0;">Cell move: <span id="cell-move" style="color:#a78bfa;">—</span></div>
        <div id="wp-efe-detail" style="margin-top:0.4rem;">
          <div class="efe-row"><label>Extrinsic</label><div class="efe-bar"><div class="efe-fill fill-extrinsic" id="wp-efe-ext" style="width:0%"></div></div></div>
          <div class="efe-row"><label>Salience</label><div class="efe-bar"><div class="efe-fill fill-salience" id="wp-efe-sal" style="width:0%"></div></div></div>
          <div class="efe-row"><label>Novelty</label><div class="efe-bar"><div class="efe-fill fill-novelty" id="wp-efe-nov" style="width:0%"></div></div></div>
        </div>
      </div>
      <div class="panel-section">
        <h3>Waypoint Policy</h3>
        <div id="wp-policy" style="max-height:200px;overflow-y:auto;"></div>
      </div>
      <div class="panel-section">
        <h3>Cumulative Reward</h3>
        <div id="reward-chart-container"><canvas id="reward-chart"></canvas></div>
      </div>
    `;
  }

  updatePanel(result: any): void {
    // Object beliefs (dynamic — works for any number of objects)
    const beliefs = result.beliefs?.target_belief ?? [];
    const ss = result.scan_state;
    const inspectedSet = new Set((ss?.objects ?? []).filter((o: any) => o.inspected).map((o: any) => o.name));
    const confirmed = result.confirmed_object;
    const trueTarget = (ss?.objects ?? []).find((o: any) => o.is_target)?.name;
    const ended = result.found_target || result.mission_failed || (result.battery ?? 999) <= 0;
    const objDiv = document.getElementById('obj-beliefs');
    if (objDiv) {
      objDiv.innerHTML = beliefs.map((b: any, i: number) => {
        const c = OBJ_CSS[i % OBJ_CSS.length];
        if (!b.discovered && !ended) {
          // Undiscovered: show dimmed with no position
          return `<div class="belief-bar"><label style="color:#404060">○ ${b.name} <span style="font-size:0.55rem">(not found)</span></label><div class="bar-track"><div class="bar-fill" style="width:${b.prob * 100}%;background:#404060"></div></div><span class="bar-value" style="color:#505070">${b.prob.toFixed(2)}</span></div>`;
        }
        const mark = inspectedSet.has(b.name) ? `<span style="color:${c};font-size:0.6rem" title="inspected">●</span>` : `<span style="color:#2a2a4a;font-size:0.6rem" title="not inspected">○</span>`;
        let tag = '';
        if (ended && b.name === trueTarget) tag = ' <span style="color:#22c55e;font-size:0.6rem;font-weight:700">★ TARGET</span>';
        else if (ended && b.name === confirmed && result.mission_failed) tag = ' <span style="color:#ef4444;font-size:0.6rem;font-weight:700">✗ WRONG</span>';
        return `<div class="belief-bar"><label style="color:${c}">${mark} ${b.name}${tag}</label><div class="bar-track"><div class="bar-fill" id="obj-${b.name}" style="width:${b.prob * 100}%;background:${c}"></div></div><span class="bar-value">${b.prob.toFixed(2)}</span></div>`;
      }).join('');
    }
    const icEl = document.getElementById('inspected-count');
    if (icEl) icEl.textContent = String(inspectedSet.size);
    const toEl = document.getElementById('total-objects');
    if (toEl) toEl.textContent = String(beliefs.length);
    const dcEl = document.getElementById('discovered-count');
    if (dcEl) dcEl.textContent = String(ss?.n_discovered ?? 0);
    const epEl = document.getElementById('explored-pct');
    if (epEl) epEl.textContent = String(ss?.explored_pct ?? 0);

    // Observation log
    const obsLog = document.getElementById('obs-log');
    if (obsLog) {
      const obs = result.observations ?? [];
      obsLog.innerHTML = obs.map((o: any) => {
        const color = o.obs === 'target-like' ? '#22c55e' : o.obs === 'distractor-like' ? '#ef4444' : '#606080';
        return `<div><span style="color:${color}">●</span> ${o.object} z=${o.altitude}: <span style="color:${color}">${o.obs}</span></div>`;
      }).join('');
    }

    // Discrimination quality
    const disc = result.beliefs?.disc_quality ?? [];
    for (const d of disc) {
      const bar = document.getElementById(`disc-z${d.altitude}`) as HTMLDivElement | null;
      const val = document.getElementById(`disc-val-z${d.altitude}`) as HTMLSpanElement | null;
      if (bar) bar.style.width = `${d.believed_accuracy * 100}%`;
      if (val) val.textContent = d.believed_accuracy.toFixed(2);
    }

    const searchProfile = result.beliefs?.search_profile ?? [];
    for (const s of searchProfile) {
      const gainBar = document.getElementById(`search-z${s.altitude}`) as HTMLDivElement | null;
      const gainVal = document.getElementById(`search-val-z${s.altitude}`) as HTMLSpanElement | null;
      const clearBar = document.getElementById(`clear-z${s.altitude}`) as HTMLDivElement | null;
      const clearVal = document.getElementById(`clear-val-z${s.altitude}`) as HTMLSpanElement | null;
      if (gainBar) gainBar.style.width = `${Math.min(s.search_gain / 3, 1) * 100}%`;
      if (gainVal) gainVal.textContent = s.search_gain.toFixed(2);
      if (clearBar) clearBar.style.width = `${s.clearance * 100}%`;
      if (clearVal) clearVal.textContent = s.clearance.toFixed(2);
    }

    const dirProfile = result.beliefs?.direction_profile ?? [];
    const dirDiv = document.getElementById('dir-profile');
    if (dirDiv) {
      dirDiv.innerHTML = dirProfile.map((entry: any) => {
        const color = entry.label === 'forward' ? '#22c55e'
          : entry.label === 'turn' ? '#3b82f6'
          : entry.label === 'reverse' ? '#ef4444'
          : entry.label === 'climb' ? '#a78bfa'
          : entry.label === 'descend' ? '#f59e0b'
          : '#06b6d4';
        const pct = Math.min(entry.value / 2, 1) * 100;
        return `<div class="belief-bar"><label style="font-size:0.7rem;color:${color}">${entry.label}</label><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div><span class="bar-value">${entry.value.toFixed(2)}</span></div>`;
      }).join('');
    }

    // Altitude
    const alt = result.position?.[2] ?? 3;
    setBar('bar-alt', 'val-alt', (alt / 4) * 100, String(alt));

    const fr = document.getElementById('fov-radius');
    if (fr) fr.textContent = String(result.scan_state?.fov_radius ?? '?');
    const ds = document.getElementById('drone-step');
    if (ds) ds.textContent = String(result.step ?? 0);
    const de = document.getElementById('drone-episode');
    if (de) de.textContent = String(result.scan_state?.episode ?? 0);

    const ts = document.getElementById('target-status');
    if (ts) {
      if (result.found_target) {
        ts.style.color = '#22c55e';
        ts.textContent = `confirmed ${result.confirmed_object} (step ${result.step})`;
      } else if (result.mission_failed) {
        ts.style.color = '#ef4444';
        ts.textContent = `WRONG — confirmed ${result.confirmed_object} (mission failed)`;
      } else if ((result.battery ?? 999) <= 0) {
        ts.style.color = '#f59e0b';
        ts.textContent = 'battery depleted — mission failed';
      } else {
        ts.style.color = '#ef4444';
        ts.textContent = 'discriminating...';
      }
    }

    // Battery
    const battMax = result.scan_state?.battery_max ?? 140;
    const batt = result.battery ?? battMax;
    const battPct = Math.max(0, (batt / battMax) * 100);
    const battColor = battPct > 40 ? '#22c55e' : battPct > 15 ? '#f59e0b' : '#ef4444';
    const battBar = document.getElementById('bar-batt') as HTMLDivElement | null;
    const battVal = document.getElementById('val-batt') as HTMLSpanElement | null;
    if (battBar) { battBar.style.width = `${battPct}%`; battBar.style.background = battColor; }
    if (battVal) battVal.textContent = `${Math.round(battPct)}%`;

    // Waypoint policy (dynamic)
    const wpPolicy = document.getElementById('wp-policy');
    if (wpPolicy) {
      const entries = Object.entries(result.policy_probs ?? {})
        .sort((a, b) => (b[1] as number) - (a[1] as number));
      wpPolicy.innerHTML = entries.map(([name, prob]) => {
        const p = prob as number;
        const color = wpColor(name);
        return `<div class="belief-bar"><label style="color:${color};font-size:0.65rem">${name}</label><div class="bar-track"><div class="bar-fill" style="width:${p * 100}%;background:${color}"></div></div><span class="bar-value">${p.toFixed(2)}</span></div>`;
      }).join('');
    }

    // Current waypoint + EFE breakdown
    const cwp = document.getElementById('current-wp');
    if (cwp) {
      const wn = result.waypoint ?? '—';
      cwp.style.color = wpColor(wn);
      cwp.textContent = wn;
    }
    const cm = document.getElementById('cell-move');
    if (cm) cm.textContent = result.action ?? '—';

    const chosenEfe = result.efe?.[result.waypoint];
    if (chosenEfe) {
      setEFE('wp-efe-ext', chosenEfe.extrinsic);
      setEFE('wp-efe-sal', chosenEfe.salience);
      setEFE('wp-efe-nov', chosenEfe.novelty);
    }

    this.rewardHistory.push(result.reward ?? 0);
    drawChart(this.rewardHistory);

    annotateEFE(result.efe, result.waypoint, {
      extrinsic: 'The drone targets a high-belief object for confirmation.',
      salience: 'The drone scans an object or explores — reducing uncertainty about the target.',
      novelty: 'The drone tests a new altitude — learning how its sensor performs at different heights.',
      tie: 'Exploration and confirmation are balanced — the drone weighs gathering evidence vs. committing.',
    });
  }

  resetPanel(): void {
    this.rewardHistory = [];
    const objDiv = document.getElementById('obj-beliefs');
    if (objDiv) objDiv.innerHTML = '';
    const obsLog = document.getElementById('obs-log');
    if (obsLog) obsLog.innerHTML = '';
    const icEl = document.getElementById('inspected-count');
    if (icEl) icEl.textContent = '0';
    const dcEl = document.getElementById('discovered-count');
    if (dcEl) dcEl.textContent = '0';
    const epEl = document.getElementById('explored-pct');
    if (epEl) epEl.textContent = '0';
    setBar('bar-alt', 'val-alt', 75, '3');
    for (const z of [2, 3, 4]) {
      setBar(`search-z${z}`, `search-val-z${z}`, 33, '1.00');
      setBar(`clear-z${z}`, `clear-val-z${z}`, 80, '0.80');
    }
    const dirDiv = document.getElementById('dir-profile');
    if (dirDiv) dirDiv.innerHTML = '';
    setEFE('wp-efe-ext', 0); setEFE('wp-efe-sal', 0); setEFE('wp-efe-nov', 0);
    const cwp = document.getElementById('current-wp');
    if (cwp) cwp.textContent = '—';
    const wpPol = document.getElementById('wp-policy');
    if (wpPol) wpPol.innerHTML = '';
  }

  // ─── 3D ───

  private makeTextSprite(text: string, color: string): THREE.Sprite {
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
    sprite.scale.set(2.0, 0.5, 1);
    return sprite;
  }

  private buildDrone(): THREE.Group {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x06b6d4, metalness: 0.7, roughness: 0.2,
      emissive: 0x06b6d4, emissiveIntensity: 0.3,
    });
    g.add(new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.3, 0.15, 8), bodyMat));
    const armMat = new THREE.MeshStandardMaterial({ color: 0x404060, metalness: 0.5, roughness: 0.3 });
    const rotorMat = new THREE.MeshStandardMaterial({ color: 0x808090, transparent: true, opacity: 0.7 });
    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
      const dist = 0.45;
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.04, dist * 2), armMat);
      arm.rotation.y = angle;
      arm.position.set(Math.cos(angle) * dist * 0.5, 0, Math.sin(angle) * dist * 0.5);
      g.add(arm);
      const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.02, 12), rotorMat);
      rotor.position.set(Math.cos(angle) * dist, 0.1, Math.sin(angle) * dist);
      rotor.userData.rotorSpeed = 0.3 + Math.random() * 0.1;
      g.add(rotor);
    }
    return g;
  }

  private buildEnvironment(ss: any) {
    // Buildings only — objects are created dynamically as they're discovered
    const bldgMat = new THREE.MeshStandardMaterial({ color: 0x2a2a4a, roughness: 0.6, metalness: 0.15 });
    for (const b of (ss.buildings ?? [])) {
      const h = b.h * CELL;
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(CELL * 0.85, h, CELL * 0.85), bldgMat);
      mesh.position.set(b.x * CELL + CELL / 2, h / 2, b.y * CELL + CELL / 2);
      mesh.castShadow = true; mesh.receiveShadow = true;
      this.scene.add(mesh);
      this.buildingMeshes.push(mesh);
    }
  }

  private updateObjectBeliefs(result: any) {
    const beliefs = result?.beliefs?.target_belief;
    const ss = result?.scan_state;
    if (!beliefs || !ss) return;
    const objects = ss.objects ?? [];

    // Create markers for newly discovered objects (or all objects on mission end)
    for (let i = 0; i < objects.length; i++) {
      const obj = objects[i];
      if (obj.x == null || this.createdObjects.has(i)) continue;

      const color = OBJ_COLORS[i % OBJ_COLORS.length];
      const mat = new THREE.MeshStandardMaterial({
        color, emissive: color, emissiveIntensity: 0.6,
        roughness: 0.25, metalness: 0.4,
      });
      const marker = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.6, 12), mat);
      marker.position.set(obj.x * CELL + CELL / 2, 0.3, obj.y * CELL + CELL / 2);
      marker.castShadow = true;
      this.scene.add(marker);
      this.objectMarkers[i] = marker;

      const label = this.makeTextSprite(obj.name, OBJ_CSS[i % OBJ_CSS.length]);
      label.position.set(obj.x * CELL + CELL / 2, 1.3, obj.y * CELL + CELL / 2);
      this.scene.add(label);
      this.objectLabels[i] = label;

      const light = new THREE.PointLight(color, 1.5, 4);
      light.position.set(obj.x * CELL + CELL / 2, 1, obj.y * CELL + CELL / 2);
      this.scene.add(light);
      this.objectLights[i] = light;

      this.createdObjects.add(i);
    }

    // Update existing markers by belief
    for (let i = 0; i < beliefs.length; i++) {
      if (!this.createdObjects.has(i)) continue;
      const prob = beliefs[i].prob;
      const marker = this.objectMarkers[i];
      if (!marker) continue;
      const mat = marker.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.2 + prob * 0.8;
      const s = 0.8 + prob * 0.6;
      marker.scale.set(s, s, s);
    }

    // Highlight on mission end: confirmed wrong → red, true target → green pulsing ring
    const ended = ss?.target_found || ss?.mission_failed || (result?.battery ?? 999) <= 0;
    if (ended) {
      for (let i = 0; i < objects.length; i++) {
        if (!this.createdObjects.has(i)) continue;
        const obj = objects[i];
        const marker = this.objectMarkers[i];
        if (!marker) continue;
        const mat = marker.material as THREE.MeshStandardMaterial;
        if (obj.is_target) {
          mat.emissive.setHex(0x22c55e);
          mat.emissiveIntensity = 1.0;
          marker.scale.set(1.6, 1.6, 1.6);
          if (this.objectLabels[i]) {
            this.scene.remove(this.objectLabels[i]);
            const tgtLabel = this.makeTextSprite(`★ ${obj.name} ★`, '#22c55e');
            tgtLabel.position.set(obj.x * CELL + CELL / 2, 1.8, obj.y * CELL + CELL / 2);
            tgtLabel.scale.set(3.0, 0.75, 1);
            this.scene.add(tgtLabel);
            this.objectLabels[i] = tgtLabel;
          }
        } else if (ss?.mission_failed && obj.name === result?.confirmed_object) {
          mat.emissive.setHex(0xef4444);
          mat.emissiveIntensity = 1.0;
          if (this.objectLabels[i]) {
            this.scene.remove(this.objectLabels[i]);
            const wrongLabel = this.makeTextSprite(`✗ ${obj.name}`, '#ef4444');
            wrongLabel.position.set(obj.x * CELL + CELL / 2, 1.8, obj.y * CELL + CELL / 2);
            wrongLabel.scale.set(3.0, 0.75, 1);
            this.scene.add(wrongLabel);
            this.objectLabels[i] = wrongLabel;
          }
        } else {
          mat.emissiveIntensity = 0.08;
          marker.scale.set(0.7, 0.7, 0.7);
        }
      }
    }
  }

  private showOutcomeOverlay(result: any): void {
    const viewport = document.getElementById('viewport');
    if (!viewport) return;
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;pointer-events:none;z-index:10;';

    let icon: string, title: string, subtitle: string, color: string, bgColor: string;
    if (result.found_target) {
      icon = '✓';
      title = `TARGET CONFIRMED — ${result.confirmed_object}`;
      subtitle = `Identified in ${result.step} steps · Reward: ${result.total_reward.toFixed(1)}`;
      color = '#22c55e';
      bgColor = 'rgba(34,197,94,0.08)';
    } else if (result.mission_failed) {
      const ss = result.scan_state;
      const trueName = (ss?.objects ?? []).find((o: any) => o.is_target)?.name ?? '?';
      icon = '✗';
      title = `WRONG TARGET — ${result.confirmed_object}`;
      subtitle = `True target was ${trueName} · Reward: ${result.total_reward.toFixed(1)}`;
      color = '#ef4444';
      bgColor = 'rgba(239,68,68,0.08)';
    } else {
      icon = '⚡';
      title = 'BATTERY DEPLETED';
      const ss = result.scan_state;
      const trueName = (ss?.objects ?? []).find((o: any) => o.is_target)?.name ?? '?';
      subtitle = `Target was ${trueName} · ${result.step} steps used`;
      color = '#f59e0b';
      bgColor = 'rgba(245,158,11,0.08)';
    }

    overlay.innerHTML = `
      <div style="background:${bgColor};backdrop-filter:blur(4px);border:2px solid ${color};border-radius:12px;padding:1.5rem 2.5rem;text-align:center;animation:fadeInScale 0.4s ease-out">
        <div style="font-size:2.5rem;color:${color};margin-bottom:0.3rem">${icon}</div>
        <div style="font-size:1.1rem;font-weight:700;color:${color};letter-spacing:0.05em">${title}</div>
        <div style="font-size:0.8rem;color:#a0a0c0;margin-top:0.3rem">${subtitle}</div>
      </div>
    `;
    viewport.style.position = 'relative';
    viewport.appendChild(overlay);
    this.outcomeOverlay = overlay;
  }

  private updateExploredTiles(cells: number[][]) {
    for (const [cx, cy] of cells) {
      const key = `${cx},${cy}`;
      if (this.exploredTiles.has(key)) continue;
      const mat = new THREE.MeshBasicMaterial({
        color: 0x06b6d4, transparent: true, opacity: 0.08, depthWrite: false,
      });
      const geo = new THREE.PlaneGeometry(CELL * 0.92, CELL * 0.92);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(cx * CELL + CELL / 2, 0.02, cy * CELL + CELL / 2);
      this.scene.add(mesh);
      this.exploredTiles.set(key, mesh);
    }
  }

  private updateFrontierHalo(ss: any, result: any) {
    for (const t of this.frontierTiles) this.scene.remove(t);
    this.frontierTiles = [];
    if (!ss || result.found_target || result.mission_failed) return;

    // Frontier = unseen cells adjacent to at least one seen cell (4-connected).
    const size: number = ss.grid_size ?? this.gridW;
    const seen: Set<string> = new Set((ss.seen_cells ?? []).map((c: number[]) => `${c[0]},${c[1]}`));
    if (seen.size === 0) return;
    const frontier: Array<[number, number]> = [];
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        if (seen.has(`${x},${y}`)) continue;
        const adj = [
          [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1],
        ];
        if (adj.some(([nx, ny]) => seen.has(`${nx},${ny}`))) {
          frontier.push([x, y]);
        }
      }
    }
    for (const [cx, cy] of frontier) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0xf59e0b, transparent: true, opacity: 0.18, depthWrite: false,
      });
      const geo = new THREE.PlaneGeometry(CELL * 0.88, CELL * 0.88);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(cx * CELL + CELL / 2, 0.05, cy * CELL + CELL / 2);
      this.scene.add(mesh);
      this.frontierTiles.push(mesh);
    }
  }

  private updateAltitudePole(pos: number[]) {
    const [px, py, pz] = pos;
    const x = px * CELL + CELL / 2;
    const y = py * CELL + CELL / 2;
    const h = pz * CELL + CELL / 2;
    if (this.altitudePole) this.scene.remove(this.altitudePole);
    const mat = new THREE.LineDashedMaterial({
      color: 0xa78bfa, dashSize: 0.25, gapSize: 0.15, transparent: true, opacity: 0.55,
    });
    const pts = [new THREE.Vector3(x, 0.05, y), new THREE.Vector3(x, h, y)];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    this.scene.add(line);
    this.altitudePole = line;

    if (!this.altitudeLabel) {
      this.altitudeLabel = this.makeTextSprite(`z=${pz}`, '#a78bfa');
      this.altitudeLabel.scale.set(1.4, 0.35, 1);
      this.scene.add(this.altitudeLabel);
    } else {
      // Regenerate label texture so altitude value stays fresh.
      this.scene.remove(this.altitudeLabel);
      this.altitudeLabel = this.makeTextSprite(`z=${pz}`, '#a78bfa');
      this.altitudeLabel.scale.set(1.4, 0.35, 1);
      this.scene.add(this.altitudeLabel);
    }
    this.altitudeLabel.position.set(x + 0.6, h / 2, y);
  }

  private updateEfeArrow(result: any) {
    if (this.efeArrow) { this.scene.remove(this.efeArrow); this.efeArrow = null; }
    const wp = result.waypoint;
    const efe = result.efe?.[wp];
    if (!wp || !efe) return;
    // Resolve waypoint target cell from scan_state's objects list when possible.
    const ss = result.scan_state;
    if (!ss) return;
    let tx: number | null = null, ty: number | null = null, tz = 2;
    for (const obj of ss.objects ?? []) {
      if (wp.includes(obj.name)) {
        if (obj.x == null) return;
        tx = obj.x; ty = obj.y;
        if (wp.startsWith('Confirm')) tz = 1;
        break;
      }
    }
    if (tx == null || ty == null) return;

    const [px, py, pz] = result.position;
    const from = new THREE.Vector3(px * CELL + CELL / 2, pz * CELL + CELL / 2, py * CELL + CELL / 2);
    const to = new THREE.Vector3(tx * CELL + CELL / 2, tz * CELL + CELL / 2, ty * CELL + CELL / 2);
    const dir = to.clone().sub(from);
    const dist = dir.length();
    if (dist < 0.1) return;
    dir.normalize();

    // Dominant component colour
    const vals: [string, number][] = [
      ['extrinsic', Math.abs(efe.extrinsic ?? 0)],
      ['salience',  Math.abs(efe.salience  ?? 0)],
      ['novelty',   Math.abs(efe.novelty   ?? 0)],
    ];
    vals.sort((a, b) => b[1] - a[1]);
    const colorHex = vals[0][0] === 'extrinsic' ? 0x22c55e
                   : vals[0][0] === 'salience'  ? 0x3b82f6
                   : 0xf59e0b;
    const arrow = new THREE.ArrowHelper(dir, from, Math.min(dist, 8), colorHex, 0.6, 0.35);
    (arrow.line as any).material.transparent = true;
    (arrow.line as any).material.opacity = 0.7;
    (arrow.cone as any).material.transparent = true;
    (arrow.cone as any).material.opacity = 0.85;
    this.scene.add(arrow);
    this.efeArrow = arrow;
  }

  private flashObservation(result: any) {
    if (this.observationFlash) { this.scene.remove(this.observationFlash); this.observationFlash = null; }
    const obs = result.observations ?? [];
    if (obs.length === 0) return;
    const latest = obs[obs.length - 1];
    if (!latest || !latest.object) return;
    // Find object position
    const ss = result.scan_state;
    if (!ss) return;
    const o = (ss.objects ?? []).find((x: any) => x.name === latest.object);
    if (!o || o.x == null) return;

    const color = latest.obs === 'target-like' ? 0x22c55e
                : latest.obs === 'distractor-like' ? 0xef4444
                : 0x3b82f6;
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.6, 1.0, 28),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(o.x * CELL + CELL / 2, 0.15, o.y * CELL + CELL / 2);
    this.scene.add(ring);
    this.observationFlash = ring;
    this.observationFlashUntil = Date.now() + 700;
  }

  private updateFrustum(pos: number[], radius: number) {
    if (this.frustumMesh) this.scene.remove(this.frustumMesh);
    const [px, py, pz] = pos;
    const r = radius * CELL;
    const h = pz * CELL;
    const mat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4, transparent: true, opacity: 0.06, side: THREE.DoubleSide,
    });
    const geo = new THREE.ConeGeometry(r, h, 16, 1, true);
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

  private onFrame() {
    const delta = this.targetPos.clone().sub(this.drone.position);
    if (delta.length() > 0.05) this.drone.position.add(delta.multiplyScalar(0.1));
    this.drone.position.y += Math.sin(Date.now() * 0.005) * 0.003;
    this.glowLight.position.copy(this.drone.position);
    for (const child of this.drone.children) {
      if (child.userData.rotorSpeed) child.rotation.y += child.userData.rotorSpeed;
    }
    // Pulse object markers (only created ones)
    for (const m of this.objectMarkers) {
      if (m) m.position.y = 0.3 + Math.sin(Date.now() * 0.003) * 0.03;
    }
    // Fade + pulse observation-flash ring
    if (this.observationFlash) {
      const mat = this.observationFlash.material as THREE.MeshBasicMaterial;
      const remaining = this.observationFlashUntil - Date.now();
      if (remaining <= 0) {
        this.scene.remove(this.observationFlash);
        this.observationFlash = null;
      } else {
        mat.opacity = Math.max(0, remaining / 700) * 0.85;
        const s = 1 + (1 - remaining / 700) * 1.5;
        this.observationFlash.scale.set(s, s, 1);
      }
    }
  }
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

const OBJ_NAME_LIST = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot'];

function wpColor(name: string): string {
  if (name.startsWith('Explore')) return '#06b6d4';
  for (let i = 0; i < OBJ_NAME_LIST.length; i++) {
    if (name.includes(OBJ_NAME_LIST[i])) return OBJ_CSS[i % OBJ_CSS.length];
  }
  return '#8080a0';
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
  const maxT = Math.max(100, history.length);
  const maxR = Math.max(Math.abs(sum), 1);
  const px = 30, py = 10, pw = canvas.width - px - 10, ph = canvas.height - py * 2;
  ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.lineTo(px + pw, py + ph); ctx.stroke();
  ctx.strokeStyle = '#06b6d4'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < cum.length; i++) {
    const x = px + (i / maxT) * pw, y = py + ph - (cum[i] / maxR) * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}
