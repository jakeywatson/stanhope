import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/*
 * Cross-shaped maze layout (top-down view):
 *
 *                  [CUE]
 *                    |
 *    [LEFT] ---- [CENTRE] ---- [RIGHT]
 *                    |
 *                 [START]
 *
 * Agent starts at south, centre is the junction.
 * Cue arm extends north. Left and right are reward arms.
 */

// World positions for each named location
const CENTRE = new THREE.Vector3(0, 0, 0);
const CUE     = new THREE.Vector3(0, 0, -10);
const LEFT    = new THREE.Vector3(-10, 0, 0);
const RIGHT   = new THREE.Vector3(10, 0, 0);
const START   = new THREE.Vector3(0, 0, 7);

export const POSITIONS: Record<string, THREE.Vector3> = {
  centre: CENTRE,
  cue: CUE,
  left: LEFT,
  right: RIGHT,
  start: START,
};

// Path waypoints: how to move from centre to each destination
export const PATHS: Record<string, THREE.Vector3[]> = {
  cue:   [CENTRE.clone(), new THREE.Vector3(0, 0, -5), CUE.clone()],
  left:  [CENTRE.clone(), new THREE.Vector3(-5, 0, 0), LEFT.clone()],
  right: [CENTRE.clone(), new THREE.Vector3(5, 0, 0), RIGHT.clone()],
  start: [CENTRE.clone(), new THREE.Vector3(0, 0, 3.5), START.clone()],
};

export function createScene(container: HTMLElement) {
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

  // Camera — elevated isometric-ish view
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 300);
  camera.position.set(18, 22, 22);
  camera.lookAt(0, 0, -1);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.target.set(0, 0, -1);
  controls.maxPolarAngle = Math.PI / 2.1;
  controls.minDistance = 10;
  controls.maxDistance = 60;

  // --- Lighting ---
  scene.add(new THREE.AmbientLight(0x303050, 0.5));

  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(10, 20, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 50;
  const sc = 20;
  sun.shadow.camera.left = -sc;
  sun.shadow.camera.right = sc;
  sun.shadow.camera.top = sc;
  sun.shadow.camera.bottom = -sc;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0x2040a0, 0.3);
  fill.position.set(-8, 10, -5);
  scene.add(fill);

  // --- Ground ---
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ color: 0x0c0c18, roughness: 0.95 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.1;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(60, 60, 0x151528, 0x151528);
  grid.position.y = -0.08;
  scene.add(grid);

  // --- Build the cross maze ---
  buildCrossMaze(scene);

  return { scene, camera, renderer, controls };
}

// ─── Materials ───

function wallMaterial() {
  return new THREE.MeshStandardMaterial({ color: 0x28284a, roughness: 0.5, metalness: 0.15 });
}
function floorMaterial() {
  return new THREE.MeshStandardMaterial({ color: 0x181830, roughness: 0.85, metalness: 0.05 });
}
function cueMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0x3b82f6, roughness: 0.35, metalness: 0.3,
    emissive: 0x3b82f6, emissiveIntensity: 0.2,
  });
}
function leftArmMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0x22c55e, roughness: 0.35, metalness: 0.3,
    emissive: 0x22c55e, emissiveIntensity: 0.15,
  });
}
function rightArmMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0xf59e0b, roughness: 0.35, metalness: 0.3,
    emissive: 0xf59e0b, emissiveIntensity: 0.15,
  });
}

// ─── Maze Construction ───

const CW = 3;        // corridor width
const WH = 2.0;      // wall height
const WT = 0.18;     // wall thickness

function buildCrossMaze(scene: THREE.Scene) {
  const fm = floorMaterial();
  const wm = wallMaterial();

  // Floors — four corridors radiating from centre
  addBox(scene, fm, CW, 0.12, 8, 0, -0.04, -5);        // north (to cue)
  addBox(scene, fm, CW, 0.12, 5, 0, -0.04, 3.5);       // south (to start)
  addBox(scene, fm, 8, 0.12, CW, -5, -0.04, 0);        // west (to left)
  addBox(scene, fm, 8, 0.12, CW, 5, -0.04, 0);         // east (to right)
  addBox(scene, fm, CW + 2, 0.12, CW + 2, 0, -0.04, 0); // centre hub

  // Room platforms
  addBox(scene, cueMaterial(), 3.5, 0.18, 3.5, 0, 0.02, -10);
  addBox(scene, leftArmMaterial(), 3.5, 0.18, 3.5, -10, 0.02, 0);
  addBox(scene, rightArmMaterial(), 3.5, 0.18, 3.5, 10, 0.02, 0);
  const smMat = new THREE.MeshStandardMaterial({
    color: 0x7c3aed, roughness: 0.35, metalness: 0.3,
    emissive: 0x7c3aed, emissiveIntensity: 0.15,
  });
  addBox(scene, smMat, 3, 0.18, 3, 0, 0.02, 7);

  // --- Walls ---
  // North corridor
  addBox(scene, wm, WT, WH, 8, -CW / 2, WH / 2, -5);
  addBox(scene, wm, WT, WH, 8, CW / 2, WH / 2, -5);
  // North room
  addBox(scene, wm, 4, WH, WT, 0, WH / 2, -11.75);
  addBox(scene, wm, WT, WH, 3.5, -2, WH / 2, -10);
  addBox(scene, wm, WT, WH, 3.5, 2, WH / 2, -10);

  // South corridor
  addBox(scene, wm, WT, WH, 5, -CW / 2, WH / 2, 3.5);
  addBox(scene, wm, WT, WH, 5, CW / 2, WH / 2, 3.5);
  // South room
  addBox(scene, wm, 3, WH, WT, 0, WH / 2, 8.5);
  addBox(scene, wm, WT, WH, 3, -1.5, WH / 2, 7);
  addBox(scene, wm, WT, WH, 3, 1.5, WH / 2, 7);

  // West corridor
  addBox(scene, wm, 8, WH, WT, -5, WH / 2, -CW / 2);
  addBox(scene, wm, 8, WH, WT, -5, WH / 2, CW / 2);
  // West room
  addBox(scene, wm, WT, WH, 4, -11.75, WH / 2, 0);
  addBox(scene, wm, 3.5, WH, WT, -10, WH / 2, -2);
  addBox(scene, wm, 3.5, WH, WT, -10, WH / 2, 2);

  // East corridor
  addBox(scene, wm, 8, WH, WT, 5, WH / 2, -CW / 2);
  addBox(scene, wm, 8, WH, WT, 5, WH / 2, CW / 2);
  // East room
  addBox(scene, wm, WT, WH, 4, 11.75, WH / 2, 0);
  addBox(scene, wm, 3.5, WH, WT, 10, WH / 2, -2);
  addBox(scene, wm, 3.5, WH, WT, 10, WH / 2, 2);

  // --- Room features ---

  // Cue tower (beacon in north room)
  const tower = new THREE.Mesh(
    new THREE.CylinderGeometry(0.3, 0.5, 3, 8),
    cueMaterial(),
  );
  tower.position.set(0, 1.5, -10);
  tower.castShadow = true;
  scene.add(tower);
  scene.add(new THREE.PointLight(0x3b82f6, 3, 8).translateY(3.2).translateZ(-10));

  // Left reward pedestal
  addPedestal(scene, leftArmMaterial(), -10, 0);
  scene.add(new THREE.PointLight(0x22c55e, 2, 6).translateX(-10).translateY(2.5));

  // Right reward pedestal
  addPedestal(scene, rightArmMaterial(), 10, 0);
  scene.add(new THREE.PointLight(0xf59e0b, 2, 6).translateX(10).translateY(2.5));

  // Start marker ring
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.7, 0.95, 32),
    new THREE.MeshBasicMaterial({ color: 0x7c3aed, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(0, 0.03, 7);
  scene.add(ring);

  // Labels
  addLabel(scene, 'CUE', 0, 3.8, -10, 0x3b82f6);
  addLabel(scene, 'LEFT', -10, 3.2, 0, 0x22c55e);
  addLabel(scene, 'RIGHT', 10, 3.2, 0, 0xf59e0b);
  addLabel(scene, 'START', 0, 2.8, 7, 0x7c3aed);
}

function addBox(
  scene: THREE.Scene, mat: THREE.Material,
  w: number, h: number, d: number,
  x: number, y: number, z: number,
) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
}

function addPedestal(scene: THREE.Scene, mat: THREE.Material, x: number, z: number) {
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.7, 0.8, 12), mat);
  base.position.set(x, 0.4, z);
  base.castShadow = true;
  scene.add(base);

  const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.35, 0), mat);
  gem.position.set(x, 1.2, z);
  gem.castShadow = true;
  scene.add(gem);
}

function addLabel(
  scene: THREE.Scene, text: string,
  x: number, y: number, z: number, color: number,
) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
  ctx.font = 'bold 32px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);

  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true }),
  );
  sprite.position.set(x, y, z);
  sprite.scale.set(3, 0.75, 1);
  scene.add(sprite);
}
