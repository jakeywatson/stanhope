import * as THREE from 'three';
import { POSITIONS, PATHS } from './MazeWorld';

export class AgentAvatar {
  mesh: THREE.Mesh;
  private glowLight: THREE.PointLight;

  // waypoint queue for path-following
  private waypoints: THREE.Vector3[] = [];
  private currentTarget: THREE.Vector3;
  private _moving = false;
  private _onArrive: (() => void) | null = null;

  constructor(scene: THREE.Scene) {
    const geo = new THREE.SphereGeometry(0.4, 32, 32);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x7c3aed,
      emissive: 0x7c3aed,
      emissiveIntensity: 0.5,
      roughness: 0.2,
      metalness: 0.6,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.position.copy(POSITIONS.centre);
    this.mesh.position.y = 0.5;
    this.mesh.castShadow = true;
    scene.add(this.mesh);

    this.glowLight = new THREE.PointLight(0x7c3aed, 2, 5);
    this.glowLight.position.copy(this.mesh.position);
    scene.add(this.glowLight);

    this.currentTarget = this.mesh.position.clone();
  }

  /** Move along a named path (centre→location) and resolve when arrived. */
  moveToLocation(location: string): Promise<void> {
    return new Promise(resolve => {
      const path = PATHS[location];
      if (!path || path.length === 0) {
        resolve();
        return;
      }
      this.waypoints = path.map(p => {
        const v = p.clone();
        v.y = 0.5;
        return v;
      });
      this.currentTarget = this.waypoints.shift()!;
      this._moving = true;
      this._onArrive = resolve;
    });
  }

  /** Instantly snap back to centre. */
  resetToStart() {
    const c = POSITIONS.centre.clone();
    c.y = 0.5;
    this.mesh.position.copy(c);
    this.currentTarget.copy(c);
    this.waypoints = [];
    this._moving = false;
    this.glowLight.position.copy(c);
  }

  get isMoving() {
    return this._moving;
  }

  update() {
    if (this._moving) {
      const delta = this.currentTarget.clone().sub(this.mesh.position);
      const dist = delta.length();
      if (dist < 0.1) {
        // Arrived at current waypoint
        this.mesh.position.copy(this.currentTarget);
        if (this.waypoints.length > 0) {
          this.currentTarget = this.waypoints.shift()!;
        } else {
          this._moving = false;
          if (this._onArrive) {
            const cb = this._onArrive;
            this._onArrive = null;
            cb();
          }
        }
      } else {
        const speed = Math.min(0.14, dist);
        this.mesh.position.add(delta.normalize().multiplyScalar(speed));
      }
    }

    // Hover bob
    this.mesh.position.y = 0.5 + Math.sin(Date.now() * 0.003) * 0.05;
    this.glowLight.position.copy(this.mesh.position);
  }
}
