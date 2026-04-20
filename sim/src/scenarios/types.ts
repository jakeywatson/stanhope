import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/** Every scenario provides a 3D scene + knows how to animate step results. */
export interface SceneController {
  /** Set up the Three.js scene. Returns scene objects. */
  init(container: HTMLElement): SceneObjects;
  /** Animate a step result (move agent, update visuals). Resolves when done. */
  animateStep(result: any): Promise<void>;
  /** Reset visuals to initial state. */
  reset(): void;
  /** Clean up Three.js resources before switching scenario. */
  dispose(): void;
  /** Build the scenario-specific side panel HTML. */
  buildPanel(): string;
  /** Update the side panel with step result data. */
  updatePanel(result: any): void;
  /** Reset panel to default state. */
  resetPanel(): void;
}

export interface SceneObjects {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  /** Called every frame in the render loop. */
  onFrame(): void;
}

/** Shared renderer setup. */
export function createBaseRenderer(container: HTMLElement): {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
} {
  const w = container.clientWidth;
  const h = container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.85;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080810);
  scene.fog = new THREE.FogExp2(0x080810, 0.012);

  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 300);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.maxPolarAngle = Math.PI / 2.1;
  controls.minDistance = 5;
  controls.maxDistance = 80;

  return { renderer, scene, camera, controls };
}

export function addBaseLighting(scene: THREE.Scene) {
  scene.add(new THREE.AmbientLight(0x303050, 0.5));

  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(10, 20, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const s = 25;
  sun.shadow.camera.left = -s;
  sun.shadow.camera.right = s;
  sun.shadow.camera.top = s;
  sun.shadow.camera.bottom = -s;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 50;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0x2040a0, 0.3);
  fill.position.set(-8, 10, -5);
  scene.add(fill);
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Names and colours for annotation rendering. */
const COMP_META: Record<string, { label: string; color: string }> = {
  extrinsic: { label: 'Extrinsic value', color: '#22c55e' },
  salience:  { label: 'Salience',        color: '#3b82f6' },
  novelty:   { label: 'Novelty',         color: '#f59e0b' },
};

/**
 * Update the #efe-annotation element with a plain-language explanation
 * of which EFE component is driving the chosen action.
 */
export function annotateEFE(
  efe: Record<string, { extrinsic: number; salience: number; novelty: number }> | undefined,
  chosenKey: string | undefined,
  scenarioHints: Record<string, string>,
) {
  const driver = document.getElementById('ann-driver');
  const detail = document.getElementById('ann-detail');
  if (!driver || !detail || !efe || !chosenKey) return;

  const e = efe[chosenKey];
  if (!e) return;

  // Identify dominant component
  const vals: [string, number][] = [
    ['extrinsic', Math.abs(e.extrinsic)],
    ['salience', Math.abs(e.salience)],
    ['novelty', Math.abs(e.novelty)],
  ];
  vals.sort((a, b) => b[1] - a[1]);
  const top = vals[0];
  const meta = COMP_META[top[0]];

  // Detect ties (second is ≥80% of first)
  const tied = vals[1][1] >= top[1] * 0.8;

  if (tied) {
    const m2 = COMP_META[vals[1][0]];
    driver.innerHTML = `<span style="color:${meta.color}">${meta.label}</span> ≈ <span style="color:${m2.color}">${m2.label}</span>`;
    detail.textContent = scenarioHints['tie'] ?? 'Multiple components are competing — the agent is balancing objectives.';
  } else {
    driver.innerHTML = `<span style="color:${meta.color}">${meta.label}</span> dominates`;
    detail.textContent = scenarioHints[top[0]] ?? '';
  }
}
