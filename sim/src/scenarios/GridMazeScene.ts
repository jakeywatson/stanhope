import * as THREE from 'three';
import {
  type SceneController, type SceneObjects,
  createBaseRenderer, addBaseLighting, sleep, annotateEFE,
} from './types';

const CELL = 1.6;
const WALL_H = 1.4;
const FLOOR_COLOR = 0x181830;
const WALL_COLOR = 0x28284a;
const AGENT_COLOR = 0x7c3aed;
const ROOM_COLORS = [0x22c55e, 0x3b82f6, 0xf59e0b, 0xef4444]; // NE, NW, SE, SW
const INFO_COLOR = 0x06b6d4;
const ACTION_NAMES = ['north', 'south', 'east', 'west'];
const GOAL_LAYOUT: Array<{ name: string | null; label: string }> = [
  { name: 'Room NW', label: 'NW' },
  { name: 'North Informant', label: 'N' },
  { name: 'Room NE', label: 'NE' },
  { name: 'West Informant', label: 'W' },
  { name: null, label: '' },
  { name: null, label: '' },
  { name: 'Room SW', label: 'SW' },
  { name: null, label: '' },
  { name: 'Room SE', label: 'SE' },
];

export class GridMazeScene implements SceneController {
  private sceneObj!: THREE.Scene;
  private agent!: THREE.Mesh;
  private glowLight!: THREE.PointLight;
  private roomMarkers: THREE.Mesh[] = [];
  private roomLights: THREE.PointLight[] = [];
  private roomLabels: THREE.Sprite[] = [];
  private beliefOverlays: THREE.Mesh[] = [];
  private infoMarkers: THREE.Mesh[] = [];
  private infoLabels: THREE.Sprite[] = [];
  private trailPoints: THREE.Vector3[] = [];
  private trailLine: THREE.Line | null = null;
  private targetPos = new THREE.Vector3(0, 0.5, 0);
  private builtGrid = false;
  private cellMeshes: Map<string, THREE.Mesh> = new Map();
  private rewardHistory: number[] = [];
  private outcomeOverlay: HTMLDivElement | null = null;
  private missionEnded = false;
  private efeArrow: THREE.ArrowHelper | null = null;
  private lastMaze: any = null;

  init(container: HTMLElement): SceneObjects {
    const { renderer, scene, camera, controls } = createBaseRenderer(container);
    this.sceneObj = scene;

    const cx = (7 * CELL) / 2;
    const cz = (7 * CELL) / 2;
    camera.position.set(cx + 10, 14, cz + 10);
    camera.lookAt(cx, 0, cz);
    controls.target.set(cx, 0, cz);

    addBaseLighting(scene);

    const gSize = 7 * CELL + 8;
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(gSize, gSize),
      new THREE.MeshStandardMaterial({ color: 0x0c0c18, roughness: 0.95 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -0.05, cz);
    ground.receiveShadow = true;
    scene.add(ground);

    // Agent sphere
    const aMat = new THREE.MeshStandardMaterial({
      color: AGENT_COLOR, emissive: AGENT_COLOR, emissiveIntensity: 0.5,
      roughness: 0.2, metalness: 0.6,
    });
    this.agent = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 24), aMat);
    const hub = this.gridToWorld(3, 3);
    this.agent.position.set(hub.x, 0.5, hub.z);
    this.agent.castShadow = true;
    scene.add(this.agent);

    this.glowLight = new THREE.PointLight(AGENT_COLOR, 2, 4);
    this.glowLight.position.copy(this.agent.position);
    scene.add(this.glowLight);

    // Trail
    const trailGeo = new THREE.BufferGeometry();
    const trailMat = new THREE.LineBasicMaterial({ color: AGENT_COLOR, transparent: true, opacity: 0.4 });
    this.trailLine = new THREE.Line(trailGeo, trailMat);
    scene.add(this.trailLine);

    return { scene, camera, renderer, controls, onFrame: () => this.onFrame() };
  }

  async animateStep(result: any): Promise<void> {
    if (!this.builtGrid && result.maze) this.buildMaze(result.maze);
    if (result.maze) this.lastMaze = result.maze;
    this.updateBeliefs(result.beliefs?.target_belief, result.maze?.rooms);
    const [r, c] = result.position;
    const wp = this.gridToWorld(r, c);
    this.targetPos.set(wp.x, 0.5, wp.z);
    await sleep(150);
    this.trailPoints.push(new THREE.Vector3(wp.x, 0.15, wp.z));
    this.updateTrail();
    this.updateOutcomeVisuals(result);
    this.updateEfeArrow(result);
    if (!this.missionEnded && (result.found_target || result.mission_failed)) {
      this.missionEnded = true;
      this.showOutcomeOverlay(result);
    }
  }

  reset(): void {
    for (const m of this.cellMeshes.values()) this.sceneObj.remove(m);
    this.cellMeshes.clear();
    for (const m of this.roomMarkers) this.sceneObj.remove(m);
    this.roomMarkers = [];
    for (const light of this.roomLights) this.sceneObj.remove(light);
    this.roomLights = [];
    for (const label of this.roomLabels) this.sceneObj.remove(label);
    this.roomLabels = [];
    for (const m of this.beliefOverlays) this.sceneObj.remove(m);
    this.beliefOverlays = [];
    for (const m of this.infoMarkers) this.sceneObj.remove(m);
    this.infoMarkers = [];
    for (const label of this.infoLabels) this.sceneObj.remove(label);
    this.infoLabels = [];
    if (this.efeArrow) { this.sceneObj.remove(this.efeArrow); this.efeArrow = null; }
    this.builtGrid = false;
    this.trailPoints = [];
    this.updateTrail();
    const hub = this.gridToWorld(3, 3);
    this.agent.position.set(hub.x, 0.5, hub.z);
    this.targetPos.copy(this.agent.position);
    this.rewardHistory = [];
    this.missionEnded = false;
    this.lastMaze = null;
    if (this.outcomeOverlay) { this.outcomeOverlay.remove(); this.outcomeOverlay = null; }
  }

  dispose(): void {}

  // ─── Panel ───

  buildPanel(): string {
    const ROOM_NAMES = ['NE', 'NW', 'SE', 'SW'];
    const ROOM_CSS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444'];
    return `
      <details class="theory-card">
        <summary>Paper Connection — Room Search</summary>
        <div class="tc-body">
          <p>The agent searches <strong>4 corner rooms</strong> for a hidden target, with 2 <strong>informant posts</strong> that give noisy directional cues.</p>
          <div class="tc-eq">G(π) = <span class="tc-ext">Extrinsic</span> (room reward) + <span class="tc-sal">Salience</span> (informant cues) + <span class="tc-nov">Novelty</span> (informant reliability)</div>
          <p><span class="tc-label tc-sal">Salience</span> drives the agent to visit informants — each cue reduces entropy about which room holds the target.</p>
          <p><span class="tc-label tc-nov">Novelty</span> drives repeat visits — the agent learns how reliable each informant is (Dirichlet learning).</p>
          <p><span class="tc-label tc-ext">Extrinsic</span> takes over once the agent is confident — it goes straight to the likely room.</p>
          <p>Entering the wrong room is a <strong>terminal mistake</strong>, so blind room checking is risky and informants matter.</p>
        </div>
      </details>
      <div class="efe-annotation" id="efe-annotation">
        <div class="ann-driver" id="ann-driver">Awaiting first step...</div>
        <div id="ann-detail"></div>
      </div>
      <div class="panel-section">
        <h3>Room Beliefs — P(target)</h3>
        ${ROOM_NAMES.map((n, i) => `
          <div class="belief-bar"><label id="room-label-${n}" style="color:${ROOM_CSS[i]}">${n}</label><div class="bar-track"><div class="bar-fill" id="room-${n}" style="width:25%;background:${ROOM_CSS[i]}"></div></div><span class="bar-value" id="room-val-${n}">0.25</span></div>
        `).join('')}
      </div>
      <div class="panel-section">
        <h3>Mission Status</h3>
        <div style="font-size:0.78rem;color:#a0a0c0;line-height:1.6;">
          <div>Status: <span id="target-status" style="color:#ef4444">searching...</span></div>
          <div>Latest observation: <span id="latest-observation" style="color:#c0c0d8">neutral</span></div>
          <div>True target: <span id="true-target-room" style="color:#606080">hidden</span></div>
        </div>
      </div>
      <div class="panel-section">
        <h3>Informants</h3>
        ${[
          { id: 'a', label: 'North Post', color: '#06b6d4' },
          { id: 'b', label: 'West Post', color: '#38bdf8' },
        ].map(info => `
          <div style="font-size:0.74rem;color:#a0a0c0;margin-bottom:0.55rem;">
            <div style="font-weight:600;color:${info.color};margin-bottom:0.2rem;">${info.label}</div>
            <div class="belief-bar"><label style="width:80px">Belief</label><div class="bar-track"><div class="bar-fill fill-salience" id="info-${info.id}-belief" style="width:60%"></div></div><span class="bar-value" id="info-${info.id}-belief-val">0.60</span></div>
            <div>Visits: <span id="info-${info.id}-visits" style="color:${info.color}">0</span> · Last cue: <span id="info-${info.id}-cue">—</span></div>
            <div>Actual reliability: <span id="info-${info.id}-truth" style="color:#606080">hidden</span></div>
          </div>
        `).join('')}
        <div style="font-size:0.68rem;color:#8080a0;">
          The bars show the agent's learned trust in each informant. Ground-truth reliability is revealed only when the run ends.
        </div>
      </div>
      <div class="panel-section">
        <h3>Local Move Policy</h3>
        ${ACTION_NAMES.map((a, i) => {
          const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b'];
          return `<div class="belief-bar"><label style="color:${colors[i]}">${a.charAt(0).toUpperCase() + a.slice(1)}</label><div class="bar-track"><div class="bar-fill" id="pol-${a}" style="width:25%;background:${colors[i]}"></div></div><span class="bar-value" id="pol-val-${a}">0.25</span></div>`;
        }).join('')}
        <div style="font-size:0.7rem;color:#8080a0;margin-top:0.3rem;">Chose: <span id="chosen-action">—</span></div>
        <div style="font-size:0.7rem;color:#8080a0;">Goal: <span id="chosen-goal">—</span></div>
      </div>
      <div class="panel-section">
        <h3>Strategic Destination EFE</h3>
        <div id="goal-heatmap" style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:0.55rem;"></div>
        <div style="font-size:0.68rem;color:#8080a0;line-height:1.5;">
          Only known strategic destinations are scored here: the 4 rooms and 2 informant posts. This is not a per-cell EFE map.
        </div>
        <div style="font-size:0.68rem;color:#8080a0;margin-top:0.35rem;">Selected goal value: <span id="goal-efe-total" style="color:#c0c0d8">—</span></div>
      </div>
      <div class="panel-section">
        <h3>Cumulative Reward</h3>
        <div id="reward-chart-container"><canvas id="reward-chart"></canvas></div>
      </div>
    `;
  }

  updatePanel(result: any): void {
    const ROOM_NAMES = ['NE', 'NW', 'SE', 'SW'];
    const beliefs = result.beliefs?.target_belief ?? [];
    const ended = !!(result.found_target || result.mission_failed);
    const trueTarget = result.target_room ?? result.maze?.target_room ?? null;
    for (const b of beliefs) {
      const bar = document.getElementById(`room-${b.room}`) as HTMLDivElement | null;
      const val = document.getElementById(`room-val-${b.room}`) as HTMLSpanElement | null;
      const label = document.getElementById(`room-label-${b.room}`) as HTMLLabelElement | null;
      if (bar) bar.style.width = `${b.prob * 100}%`;
      if (val) val.textContent = b.prob.toFixed(2);
      if (label) {
        const roomColor = label.style.color;
        if (ended && b.room === trueTarget) {
          const tag = result.found_target ? 'FOUND' : 'TRUE TARGET';
          label.innerHTML = `${b.room} <span style="color:${roomColor};font-size:0.58rem;font-weight:700">★ ${tag}</span>`;
        } else {
          label.textContent = b.room;
        }
      }
    }

    const informants = result.informants ?? [];
    for (const [index, info] of informants.entries()) {
      const suffix = index === 0 ? 'a' : 'b';
      const visitsEl = document.getElementById(`info-${suffix}-visits`);
      const cueEl = document.getElementById(`info-${suffix}-cue`);
      const truthEl = document.getElementById(`info-${suffix}-truth`);
      const beliefBar = document.getElementById(`info-${suffix}-belief`) as HTMLDivElement | null;
      const beliefVal = document.getElementById(`info-${suffix}-belief-val`);
      if (visitsEl) visitsEl.textContent = String(info.visits ?? 0);
      if (cueEl) cueEl.textContent = info.last_cue ?? '—';
      if (beliefBar) beliefBar.style.width = `${(info.believed_accuracy ?? 0) * 100}%`;
      if (beliefVal) beliefVal.textContent = (info.believed_accuracy ?? 0).toFixed(2);
      if (truthEl) {
        if (typeof info.true_reliability === 'number') {
          truthEl.textContent = `${Math.round(info.true_reliability * 100)}%`;
          (truthEl as HTMLSpanElement).style.color = '#f8fafc';
        } else {
          truthEl.textContent = 'hidden';
          (truthEl as HTMLSpanElement).style.color = '#606080';
        }
      }
    }

    const ts = document.getElementById('target-status');
    const latestObs = document.getElementById('latest-observation');
    const truthEl = document.getElementById('true-target-room');
    if (ts) {
      if (result.found_target) {
        ts.style.color = '#22c55e';
        ts.textContent = `found ${result.target_room} in ${result.step} steps`;
      } else if (result.mission_failed && result.failure_reason === 'wrong_room') {
        ts.style.color = '#ef4444';
        ts.textContent = `wrong room entered — target was ${result.target_room}`;
      } else if (result.mission_failed) {
        ts.style.color = '#f59e0b';
        ts.textContent = `step cap reached — target was ${result.target_room}`;
      } else {
        ts.style.color = '#ef4444';
        ts.textContent = 'searching — target hidden';
      }
    }
    if (latestObs) latestObs.textContent = result.observation_name ?? 'neutral';
    if (truthEl) {
      if (ended && trueTarget) {
        truthEl.textContent = trueTarget;
        truthEl.style.color = result.found_target ? '#22c55e' : '#f8fafc';
      } else {
        truthEl.textContent = 'hidden';
        truthEl.style.color = '#606080';
      }
    }

    const colors: Record<string, string> = { north: '#3b82f6', south: '#ef4444', east: '#22c55e', west: '#f59e0b' };
    for (const a of ACTION_NAMES) {
      const pe = document.getElementById(`pol-${a}`) as HTMLDivElement | null;
      if (pe) pe.style.width = `${(result.policy_probs[a] ?? 0) * 100}%`;
      const ve = document.getElementById(`pol-val-${a}`) as HTMLSpanElement | null;
      if (ve) ve.textContent = (result.policy_probs[a] ?? 0).toFixed(2);
    }
    const ca = document.getElementById('chosen-action');
    if (ca) {
      const c = colors[result.action] ?? '#fff';
      ca.innerHTML = `<span style="color:${c}">${result.action}</span>`;
    }
    const cg = document.getElementById('chosen-goal');
    if (cg) {
      const goalColor = result.goal_type === 'informant' ? '#06b6d4' : '#f8fafc';
      cg.innerHTML = `<span style="color:${goalColor}">${result.goal ?? '—'}</span>`;
    }

    this.updateGoalHeatmap(result.goal_efe ?? {}, result.goal);

    this.rewardHistory.push(result.reward);
    drawChart(this.rewardHistory);

    annotateEFE(result.goal_efe, result.goal, {
      extrinsic: 'The agent commits to a room because it currently expects that destination to pay off best.',
      salience: 'The agent commits to an informant because that location is expected to reduce room uncertainty most.',
      novelty: 'The agent re-checks an informant because learning its reliability still has value.',
      tie: 'Room exploitation and information gathering are closely balanced at the strategic level.',
    });
  }

  resetPanel(): void {
    this.rewardHistory = [];
    const ROOM_NAMES = ['NE', 'NW', 'SE', 'SW'];
    for (const n of ROOM_NAMES) {
      const bar = document.getElementById(`room-${n}`) as HTMLDivElement | null;
      if (bar) bar.style.width = '25%';
      const val = document.getElementById(`room-val-${n}`) as HTMLSpanElement | null;
      if (val) val.textContent = '0.25';
      const label = document.getElementById(`room-label-${n}`) as HTMLLabelElement | null;
      if (label) label.textContent = n;
    }
    const targetStatus = document.getElementById('target-status');
    if (targetStatus) {
      targetStatus.textContent = 'searching...';
      (targetStatus as HTMLSpanElement).style.color = '#ef4444';
    }
    const latestObs = document.getElementById('latest-observation');
    if (latestObs) latestObs.textContent = 'neutral';
    const truthEl = document.getElementById('true-target-room');
    if (truthEl) {
      truthEl.textContent = 'hidden';
      (truthEl as HTMLSpanElement).style.color = '#606080';
    }
    for (const suffix of ['a', 'b']) {
      const visits = document.getElementById(`info-${suffix}-visits`);
      const cue = document.getElementById(`info-${suffix}-cue`);
      const truth = document.getElementById(`info-${suffix}-truth`);
      const belief = document.getElementById(`info-${suffix}-belief`) as HTMLDivElement | null;
      const beliefVal = document.getElementById(`info-${suffix}-belief-val`);
      if (visits) visits.textContent = '0';
      if (cue) cue.textContent = '—';
      if (truth) {
        truth.textContent = 'hidden';
        (truth as HTMLSpanElement).style.color = '#606080';
      }
      if (belief) belief.style.width = '60%';
      if (beliefVal) beliefVal.textContent = '0.60';
    }
    this.updateGoalHeatmap({}, undefined);
  }

  // ─── 3D ───

  private gridToWorld(row: number, col: number) {
    return { x: col * CELL + CELL / 2, z: row * CELL + CELL / 2 };
  }

  private makeTextSprite(text: string, color: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    ctx.font = 'bold 32px sans-serif';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.8, 0.45, 1);
    return sprite;
  }

  private buildMaze(maze: any) {
    this.builtGrid = true;
    const wallMat = new THREE.MeshStandardMaterial({ color: WALL_COLOR, roughness: 0.5, metalness: 0.15 });
    const floorMat = new THREE.MeshStandardMaterial({ color: FLOOR_COLOR, roughness: 0.85 });

    const grid: number[][] = maze.grid;
    for (let r = 0; r < maze.rows; r++) {
      for (let c = 0; c < maze.cols; c++) {
        const wp = this.gridToWorld(r, c);
        if (grid[r][c] === 1) {
          const wall = new THREE.Mesh(new THREE.BoxGeometry(CELL * 0.95, WALL_H, CELL * 0.95), wallMat);
          wall.position.set(wp.x, WALL_H / 2, wp.z);
          wall.castShadow = true; wall.receiveShadow = true;
          this.sceneObj.add(wall);
          this.cellMeshes.set(`${r},${c}`, wall);
        } else {
          const floor = new THREE.Mesh(new THREE.BoxGeometry(CELL * 0.95, 0.08, CELL * 0.95), floorMat);
          floor.position.set(wp.x, 0.04, wp.z);
          floor.receiveShadow = true;
          this.sceneObj.add(floor);
          this.cellMeshes.set(`${r},${c}`, floor);
        }
      }
    }

    // Room markers (corner glowing diamonds)
    for (let i = 0; i < maze.rooms.length; i++) {
      const rm = maze.rooms[i];
      const wp = this.gridToWorld(rm.pos[0], rm.pos[1]);
      const mat = new THREE.MeshStandardMaterial({
        color: ROOM_COLORS[i], emissive: ROOM_COLORS[i], emissiveIntensity: 0.4,
        roughness: 0.2, metalness: 0.5,
      });
      const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.3, 0), mat);
      marker.position.set(wp.x, 0.6, wp.z);
      this.sceneObj.add(marker);
      this.roomMarkers.push(marker);
      const light = new THREE.PointLight(ROOM_COLORS[i], 1.5, 4);
      light.position.set(wp.x, 1.2, wp.z);
      this.sceneObj.add(light);
      this.roomLights.push(light);

      const label = this.makeTextSprite(rm.name, '#' + ROOM_COLORS[i].toString(16).padStart(6, '0'));
      label.position.set(wp.x, 1.35, wp.z);
      this.sceneObj.add(label);
      this.roomLabels.push(label);
    }

    // Informant markers (cyan pillars)
    for (const inf of maze.informants) {
      const wp = this.gridToWorld(inf.pos[0], inf.pos[1]);
      const mat = new THREE.MeshStandardMaterial({
        color: INFO_COLOR, emissive: INFO_COLOR, emissiveIntensity: 0.5,
        roughness: 0.3, metalness: 0.4,
      });
      const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.8, 8), mat);
      pillar.position.set(wp.x, 0.4, wp.z);
      this.sceneObj.add(pillar);
      this.infoMarkers.push(pillar);

      const label = this.makeTextSprite(inf.name, '#06b6d4');
      label.position.set(wp.x, 1.15, wp.z);
      this.sceneObj.add(label);
      this.infoLabels.push(label);
    }
  }

  private updateBeliefs(beliefs: any[], rooms: any[]) {
    // Remove old overlays
    for (const m of this.beliefOverlays) this.sceneObj.remove(m);
    this.beliefOverlays = [];
    if (!beliefs || !rooms) return;

    for (let i = 0; i < beliefs.length; i++) {
      const b = beliefs[i];
      const rp = b.position ?? rooms[i]?.pos;
      if (!rp) continue;
      const wp = this.gridToWorld(rp[0], rp[1]);
      const intensity = Math.min(b.prob * 3, 1);
      const mat = new THREE.MeshBasicMaterial({
        color: ROOM_COLORS[i], transparent: true, opacity: intensity * 0.35,
      });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(CELL * 1.5, 0.02, CELL * 1.5), mat);
      mesh.position.set(wp.x, 0.1, wp.z);
      this.sceneObj.add(mesh);
      this.beliefOverlays.push(mesh);
    }
  }

  private updateOutcomeVisuals(result: any): void {
    const rooms = result.maze?.rooms ?? [];
    const ended = !!(result.found_target || result.mission_failed);
    if (!ended) return;

    for (let i = 0; i < rooms.length; i++) {
      const marker = this.roomMarkers[i];
      const light = this.roomLights[i];
      if (!marker || !light) continue;
      const mat = marker.material as THREE.MeshStandardMaterial;
      if (rooms[i]?.is_target) {
        mat.emissiveIntensity = 1.0;
        marker.scale.set(1.8, 1.8, 1.8);
        light.intensity = 3.0;
        if (this.roomLabels[i]) {
          this.sceneObj.remove(this.roomLabels[i]);
          const label = this.makeTextSprite(`★ ${rooms[i].name} ★`, '#22c55e');
          const wp = this.gridToWorld(rooms[i].pos[0], rooms[i].pos[1]);
          label.position.set(wp.x, 1.6, wp.z);
          label.scale.set(2.6, 0.6, 1);
          this.sceneObj.add(label);
          this.roomLabels[i] = label;
        }
      } else {
        mat.emissiveIntensity = 0.15;
        marker.scale.set(0.9, 0.9, 0.9);
        light.intensity = 0.9;
      }
    }
  }

  private updateEfeArrow(result: any) {
    if (this.efeArrow) { this.sceneObj.remove(this.efeArrow); this.efeArrow = null; }
    const goal = result.goal;
    const efe = result.goal_efe?.[goal];
    const maze = result.maze ?? this.lastMaze;
    if (!goal || !efe || !maze || this.missionEnded) return;
    // Resolve goal cell position
    let gr: number | null = null, gc: number | null = null;
    for (const room of maze.rooms ?? []) {
      if (room.name === goal) { gr = room.pos[0]; gc = room.pos[1]; break; }
    }
    if (gr == null) {
      for (const inf of maze.informants ?? []) {
        if (inf.name === goal) { gr = inf.pos[0]; gc = inf.pos[1]; break; }
      }
    }
    if (gr == null || gc == null) return;

    const [ar, ac] = result.position;
    const from = this.gridToWorld(ar, ac);
    const to = this.gridToWorld(gr, gc);
    const dir = new THREE.Vector3(to.x - from.x, 0, to.z - from.z);
    const dist = dir.length();
    if (dist < 0.1) return;
    dir.normalize();

    const vals: [string, number][] = [
      ['extrinsic', Math.abs(efe.extrinsic ?? 0)],
      ['salience',  Math.abs(efe.salience  ?? 0)],
      ['novelty',   Math.abs(efe.novelty   ?? 0)],
    ];
    vals.sort((a, b) => b[1] - a[1]);
    const colorHex = vals[0][0] === 'extrinsic' ? 0x22c55e
                   : vals[0][0] === 'salience'  ? 0x3b82f6
                   : 0xf59e0b;
    const origin = new THREE.Vector3(from.x, 0.9, from.z);
    const arrow = new THREE.ArrowHelper(dir, origin, Math.min(dist, 5), colorHex, 0.5, 0.28);
    (arrow.line as any).material.transparent = true;
    (arrow.line as any).material.opacity = 0.75;
    (arrow.cone as any).material.transparent = true;
    (arrow.cone as any).material.opacity = 0.9;
    this.sceneObj.add(arrow);
    this.efeArrow = arrow;
  }

  private updateGoalHeatmap(goalEFE: Record<string, { extrinsic: number; salience: number; novelty: number; total: number }>, chosenGoal: string | undefined): void {
    const container = document.getElementById('goal-heatmap');
    const totalEl = document.getElementById('goal-efe-total');
    if (!container) return;

    const totals = Object.values(goalEFE).map(entry => entry.total ?? 0);
    const maxAbs = Math.max(1, ...totals.map(value => Math.abs(value)));
    const cells = GOAL_LAYOUT.map(({ name, label }) => {
      if (!name) {
        return '<div style="height:34px;border-radius:6px;background:transparent;border:1px dashed rgba(38,38,62,0.35);"></div>';
      }

      const entry = goalEFE[name];
      const total = entry?.total ?? 0;
      const ratio = Math.min(Math.abs(total) / maxAbs, 1);
      const bg = total >= 0
        ? `rgba(59,130,246,${0.14 + ratio * 0.58})`
        : `rgba(239,68,68,${0.14 + ratio * 0.58})`;
      const border = name === chosenGoal ? '#f8fafc' : '#26263e';
      const title = `${name} | total=${total.toFixed(2)} | ext=${(entry?.extrinsic ?? 0).toFixed(2)} sal=${(entry?.salience ?? 0).toFixed(2)} nov=${(entry?.novelty ?? 0).toFixed(2)}`;
      return `<div title="${title}" style="height:34px;border-radius:6px;border:1px solid ${border};background:${bg};display:flex;align-items:center;justify-content:center;font-size:0.66rem;font-weight:700;color:#f8fafc;box-shadow:${name === chosenGoal ? '0 0 0 1px rgba(248,250,252,0.6) inset' : 'none'};">${label}</div>`;
    });
    container.innerHTML = cells.join('');
    if (totalEl) {
      const selected = chosenGoal ? goalEFE[chosenGoal] : undefined;
      totalEl.textContent = selected ? `${chosenGoal}: ${selected.total.toFixed(2)}` : '—';
    }
  }

  private showOutcomeOverlay(result: any): void {
    const viewport = document.getElementById('viewport');
    if (!viewport) return;
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;pointer-events:none;z-index:10;';

    let icon: string;
    let title: string;
    let subtitle: string;
    let color: string;
    let bgColor: string;
    if (result.found_target) {
      icon = '✓';
      title = `TARGET FOUND — ${result.target_room}`;
      subtitle = `Reached in ${result.step} steps · Reward: ${result.total_reward.toFixed(1)}`;
      color = '#22c55e';
      bgColor = 'rgba(34,197,94,0.08)';
    } else if (result.failure_reason === 'wrong_room') {
      icon = '✗';
      title = `WRONG ROOM — ${result.target_room}`;
      subtitle = `Entered the wrong room at step ${result.step} · Reward: ${result.total_reward.toFixed(1)}`;
      color = '#ef4444';
      bgColor = 'rgba(239,68,68,0.08)';
    } else {
      icon = '⌛';
      title = `SEARCH EXHAUSTED — ${result.target_room}`;
      subtitle = `Step cap reached at ${result.step} steps · Reward: ${result.total_reward.toFixed(1)}`;
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

  private updateTrail() {
    if (!this.trailLine || this.trailPoints.length < 2) return;
    const positions = new Float32Array(this.trailPoints.length * 3);
    for (let i = 0; i < this.trailPoints.length; i++) {
      positions[i * 3] = this.trailPoints[i].x;
      positions[i * 3 + 1] = 0.15;
      positions[i * 3 + 2] = this.trailPoints[i].z;
    }
    this.trailLine.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.trailLine.geometry = geo;
  }

  private onFrame() {
    const delta = this.targetPos.clone().sub(this.agent.position);
    if (delta.length() > 0.05) this.agent.position.add(delta.multiplyScalar(0.12));
    this.agent.position.y = 0.5 + Math.sin(Date.now() * 0.003) * 0.04;
    this.glowLight.position.copy(this.agent.position);
    // Spin room markers
    for (const m of this.roomMarkers) {
      m.rotation.y += 0.015;
      m.position.y = 0.6 + Math.sin(Date.now() * 0.004) * 0.08;
    }
    // Pulse informant markers
    for (const m of this.infoMarkers) {
      m.position.y = 0.4 + Math.sin(Date.now() * 0.005) * 0.05;
    }
  }
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
  const maxT = Math.max(100, history.length);
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
}
