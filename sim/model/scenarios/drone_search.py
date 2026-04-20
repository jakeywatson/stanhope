"""Aerial object discrimination scenario — tailored to Stanhope AI.

Models a drone searching an area for a specific target object among visually
similar distractors. Directly reflects Stanhope AI's FEM world model:
autonomous drones using active inference to navigate and discriminate in
uncertain environments with limited prior data.

The key mechanic: altitude controls observation quality.
  - High altitude (z=3-4): wide FOV but objects look identical (ambiguous)
  - Low altitude (z=1-2): narrow FOV but can discriminate target vs distractor
  - Descending to inspect an object = the "cue" analog from the T-maze

Hidden state: which of N objects is the real target
Beliefs: P(target = object_i) updated via Bayesian inference
Dirichlet: learns how discriminating each altitude level actually is

Active inference components:
  - Extrinsic: descend to z=1 over high-P objects for confirmation reward
  - Salience: descend near objects to discriminate (reduce hidden-state entropy)
  - Novelty: try different altitudes to learn the observation model (Dirichlet KL)

Behavioral signatures:
  - Greedy: dives toward whichever object currently has highest P
  - Active Inference: inspects multiple objects before committing (salience)
  - Active Learning: tries different altitudes to learn discrimination model (novelty)
  - Combined: survey → inspect → confirm, the optimal three-phase strategy
"""
import numpy as np
from numpy import ndarray
from math_utils import softmax, log_stable, entropy, dirichlet_expected, kl_dirichlet
from agents import AGENTS, Agent

# Grid defaults
GRID_SIZE = 12
MAX_Z = 4

# Actions
MOVE_N = 0; MOVE_S = 1; MOVE_E = 2; MOVE_W = 3
MOVE_UP = 4; MOVE_DOWN = 5; CONFIRM = 6
N_ACTIONS = 7
ACTION_NAMES = ['north', 'south', 'east', 'west', 'up', 'down', 'confirm']
ACTION_DELTAS = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0)]

# Observations
OBS_NOTHING = 0
OBS_BLOB = 1
OBS_TARGET_SIG = 2
OBS_DISTRACTOR_SIG = 3

# Object names
OBJECT_NAMES = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot']
N_OBJECTS = 6
N_BUILDINGS = 10

# True discrimination quality per altitude: P(correct | altitude z)
# Agent does NOT know these — learns via Dirichlet
# z=2 is the sweet spot (good view, proper distance to discriminate)
# z=1 is too close for the camera to focus properly (barely discriminatory)
TRUE_QUALITY = [0.50, 0.55, 0.75, 0.52, 0.50]  # z=0,1,2,3,4

# Threshold: below this, observations are ambiguous blobs
DISC_THRESHOLD = 0.53

# Battery
BATTERY_MAX = 280
BATTERY_COST_MOVE = 1
BATTERY_COST_VERT = 2  # altitude changes cost more

# Waypoint types
WP_SCAN = 'scan'
WP_CONFIRM = 'confirm'
WP_EXPLORE = 'explore'

# Max explore waypoints generated per step
N_EXPLORE_WPS = 4
EXPLORE_ALTITUDES = (2, 3, 4)

# Relative move classes for transferable search learning
DIRECTION_LABELS = ['seek', 'forward', 'turn', 'reverse', 'climb', 'descend']
DIRECTION_INDEX = {name: idx for idx, name in enumerate(DIRECTION_LABELS)}


class DroneSearchScenario:
    """Drone discriminating a target object among similar distractors."""

    def __init__(self, grid_size: int = GRID_SIZE, n_buildings: int | None = None, battery_max: int = BATTERY_MAX):
        self.agent: Agent = AGENTS['combined']
        self.grid_size = int(grid_size)
        area_scale = (self.grid_size / GRID_SIZE) ** 2
        self.n_buildings = int(n_buildings if n_buildings is not None else max(4, round(N_BUILDINGS * area_scale)))
        self.battery_max = int(battery_max)
        self.step_count = 0
        self.episode = 0
        # Sensor model persists across episodes — this is what the drone "learns"
        self.disc_conc = np.array([[1.2, 0.8]] * (MAX_Z + 1), dtype=float)
        # Search efficiency learned across episodes: expected gain from explore steps at each altitude.
        self.search_gain_sum = np.ones(MAX_Z + 1, dtype=float)
        self.search_gain_count = np.ones(MAX_Z + 1, dtype=float)
        # Blocker model learned across episodes: expected view clearance at each altitude.
        self.blocker_conc = np.array([[4.0, 1.0]] * (MAX_Z + 1), dtype=float)
        # Relative direction model learned across episodes for explore motion.
        self.dir_gain_sum = np.ones((MAX_Z + 1, len(DIRECTION_LABELS)), dtype=float)
        self.dir_gain_count = np.ones((MAX_Z + 1, len(DIRECTION_LABELS)), dtype=float)
        self._init_env()

    def _init_env(self):
        self.step_count = 0
        self.total_reward = 0.0
        self.found_target = False
        self.mission_failed = False
        self.confirmed_object = -1
        self.battery = self.battery_max
        self.info_altitude_sum = 0.0
        self.info_altitude_count = 0.0

        rng = np.random.default_rng()
        cx = self.grid_size // 2

        # Buildings (navigation obstacles, 1-cell footprint)
        building_cells = {(cx, cx)}
        self.buildings = []
        self.building_height_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for _ in range(self.n_buildings):
            for _ in range(30):
                bx = int(rng.integers(1, self.grid_size - 1))
                by = int(rng.integers(1, self.grid_size - 1))
                bh = int(rng.integers(1, 3))
                if (bx, by) not in building_cells:
                    self.buildings.append({'x': bx, 'y': by, 'h': bh})
                    building_cells.add((bx, by))
                    self.building_height_map[bx, by] = bh
                    break

        # Passability grid per altitude
        self.passable = {}
        for z in range(1, MAX_Z + 1):
            layer = np.ones((self.grid_size, self.grid_size), dtype=bool)
            for b in self.buildings:
                if z <= b['h']:
                    layer[b['x'], b['y']] = False
            self.passable[z] = layer

        # Place objects on open ground
        candidates = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                       if (x, y) not in building_cells]
        rng.shuffle(candidates)
        self.objects = [candidates[i] for i in range(min(N_OBJECTS, len(candidates)))]
        self.n_objects = len(self.objects)
        self.true_target = int(rng.integers(self.n_objects))
        self._fov_cache: dict[tuple[int, int, int], frozenset[tuple[int, int]]] = {}
        self._clearance_cache: dict[tuple[int, int, int], float] = {}
        self._visible_cache: dict[tuple[int, int, int], tuple[tuple[int, float], ...]] = {}

        # Drone starts at centre, high altitude
        self.drone_pos = [cx, cx, 3]
        self.flight_path: list[list[int]] = [list(self.drone_pos)]

        # Target beliefs: P(target = object_i)
        self.target_belief = np.ones(self.n_objects) / self.n_objects

        # disc_conc persists on self — NOT reset here

        self.obs_log: list[dict] = []
        self.inspected: set[int] = set()
        self.discovered: set[int] = set()
        self.seen_cells: set[tuple[int, int]] = set()

        # Waypoint commitment
        self.current_wp: dict | None = None
        self.wp_steps = 0  # steps committed to current waypoint
        self.prev_heading: tuple[int, int] | None = None

        # Initial scan from starting position
        x0, y0, z0 = self.drone_pos
        self.seen_cells = set(self._fov_cells(x0, y0, z0))
        for obj_idx, _dist in self._visible_objects(x0, y0, z0):
            self.discovered.add(obj_idx)

    def reset(self, agent_type: str = 'combined') -> None:
        self.agent = AGENTS.get(agent_type, AGENTS['combined'])
        self.episode += 1
        # Decay concentrations toward prior — prevents overconfidence, models sensor drift
        prior = np.array([1.2, 0.8])
        self.disc_conc = 0.8 * self.disc_conc + 0.2 * prior
        self.search_gain_sum = 0.95 * self.search_gain_sum + 0.05 * np.ones(MAX_Z + 1, dtype=float)
        self.search_gain_count = 0.95 * self.search_gain_count + 0.05 * np.ones(MAX_Z + 1, dtype=float)
        self.blocker_conc = 0.9 * self.blocker_conc + 0.1 * np.array([[4.0, 1.0]] * (MAX_Z + 1), dtype=float)
        self.dir_gain_sum = 0.95 * self.dir_gain_sum + 0.05 * np.ones((MAX_Z + 1, len(DIRECTION_LABELS)), dtype=float)
        self.dir_gain_count = 0.95 * self.dir_gain_count + 0.05 * np.ones((MAX_Z + 1, len(DIRECTION_LABELS)), dtype=float)
        self._init_env()

    def hard_reset(self, agent_type: str = 'combined') -> None:
        """Full reset including learned sensor and search models."""
        self.agent = AGENTS.get(agent_type, AGENTS['combined'])
        self.episode = 0
        self.disc_conc = np.array([[1.2, 0.8]] * (MAX_Z + 1), dtype=float)
        self.search_gain_sum = np.ones(MAX_Z + 1, dtype=float)
        self.search_gain_count = np.ones(MAX_Z + 1, dtype=float)
        self.blocker_conc = np.array([[4.0, 1.0]] * (MAX_Z + 1), dtype=float)
        self.dir_gain_sum = np.ones((MAX_Z + 1, len(DIRECTION_LABELS)), dtype=float)
        self.dir_gain_count = np.ones((MAX_Z + 1, len(DIRECTION_LABELS)), dtype=float)
        self._init_env()

    def export_learning(self, include_sensor: bool = True, include_search: bool = True) -> dict:
        """Export learned priors so they can be transferred to a new environment."""
        state = {'episodes': self.episode}
        if include_sensor:
            state['disc_conc'] = self.disc_conc.tolist()
        if include_search:
            state['search_gain_sum'] = self.search_gain_sum.tolist()
            state['search_gain_count'] = self.search_gain_count.tolist()
            state['blocker_conc'] = self.blocker_conc.tolist()
            state['dir_gain_sum'] = self.dir_gain_sum.tolist()
            state['dir_gain_count'] = self.dir_gain_count.tolist()
        return state

    def export_search_learning(self) -> dict:
        return self.export_learning(include_sensor=False, include_search=True)

    def import_learning(self, state: dict, include_sensor: bool = True, include_search: bool = True) -> None:
        """Import learned priors from another scenario instance."""
        if include_sensor and 'disc_conc' in state:
            self.disc_conc = np.array(state['disc_conc'], dtype=float)
        if include_search and 'search_gain_sum' in state:
            self.search_gain_sum = np.array(state['search_gain_sum'], dtype=float)
        if include_search and 'search_gain_count' in state:
            self.search_gain_count = np.array(state['search_gain_count'], dtype=float)
        if include_search and 'blocker_conc' in state:
            self.blocker_conc = np.array(state['blocker_conc'], dtype=float)
        if include_search and 'dir_gain_sum' in state:
            self.dir_gain_sum = np.array(state['dir_gain_sum'], dtype=float)
        if include_search and 'dir_gain_count' in state:
            self.dir_gain_count = np.array(state['dir_gain_count'], dtype=float)

    def import_search_learning(self, state: dict) -> None:
        self.import_learning(state, include_sensor=False, include_search=True)

    def _learned_search_efficiency(self, z: int) -> float:
        return float(self.search_gain_sum[z] / max(self.search_gain_count[z], 1e-8))

    def _learned_blocker_clearance(self, z: int) -> float:
        conc = self.blocker_conc[z]
        return float(conc[0] / max(conc.sum(), 1e-8))

    def _learned_direction_efficiency(self, z: int, label: str) -> float:
        idx = DIRECTION_INDEX[label]
        return float(self.dir_gain_sum[z, idx] / max(self.dir_gain_count[z, idx], 1e-8))

    def _learned_direction_bias(self, z: int, label: str) -> float:
        baseline = float(np.mean([self._learned_direction_efficiency(z, name) for name in DIRECTION_LABELS]))
        return self._learned_direction_efficiency(z, label) - baseline

    def _direction_profile(self) -> list[dict]:
        profile = []
        for label in DIRECTION_LABELS:
            sum_total = float(self.dir_gain_sum[list(EXPLORE_ALTITUDES), DIRECTION_INDEX[label]].sum())
            count_total = float(self.dir_gain_count[list(EXPLORE_ALTITUDES), DIRECTION_INDEX[label]].sum())
            profile.append({'label': label, 'value': round(sum_total / max(count_total, 1e-8), 3)})
        return profile

    def _fov_radius(self, z: int) -> int:
        return max(1, z)

    def _fov_cells(self, x: int, y: int, z: int) -> set[tuple[int, int]]:
        """Grid cells within FOV at given position."""
        key = (x, y, z)
        cached = self._fov_cache.get(key)
        if cached is not None:
            return cached

        radius = self._fov_radius(z)
        cells = set()
        for cx in range(max(0, x - radius), min(self.grid_size, x + radius + 1)):
            for cy in range(max(0, y - radius), min(self.grid_size, y + radius + 1)):
                if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
                    cells.add((cx, cy))
        frozen = frozenset(cells)
        self._fov_cache[key] = frozen
        return frozen

    def _line_cells(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps <= 1:
            return []
        cells = []
        seen = set()
        for step in range(1, steps):
            t = step / steps
            cx = int(round(x0 + (x1 - x0) * t))
            cy = int(round(y0 + (y1 - y0) * t))
            cell = (cx, cy)
            if cell not in seen and cell not in {(x0, y0), (x1, y1)}:
                cells.append(cell)
                seen.add(cell)
        return cells

    def _is_occluded(self, x: int, y: int, z: int, ox: int, oy: int) -> bool:
        for cx, cy in self._line_cells(x, y, ox, oy):
            if self.building_height_map[cx, cy] >= z:
                return True
        return False

    def _fov_clearance(self, x: int, y: int, z: int) -> float:
        key = (x, y, z)
        cached = self._clearance_cache.get(key)
        if cached is not None:
            return cached

        fov = self._fov_cells(x, y, z)
        if not fov:
            return 1.0
        blocked = sum(1 for cx, cy in fov if self.building_height_map[cx, cy] >= z)
        clearance = 1.0 - blocked / len(fov)
        self._clearance_cache[key] = clearance
        return clearance

    def _horizontal_heading(self, action: int) -> tuple[int, int] | None:
        if action not in (MOVE_N, MOVE_S, MOVE_E, MOVE_W):
            return None
        dx, dy, _ = ACTION_DELTAS[action]
        return (dx, dy)

    def _move_class(self, action: int, prev_heading: tuple[int, int] | None) -> str:
        if action == MOVE_UP:
            return 'climb'
        if action == MOVE_DOWN:
            return 'descend'
        heading = self._horizontal_heading(action)
        if heading is None or prev_heading is None:
            return 'seek'
        if heading == prev_heading:
            return 'forward'
        if heading == (-prev_heading[0], -prev_heading[1]):
            return 'reverse'
        return 'turn'

    def _visible_objects(self, x: int, y: int, z: int) -> list[tuple[int, float]]:
        """Return (object_idx, distance) for objects within FOV."""
        key = (x, y, z)
        cached = self._visible_cache.get(key)
        if cached is not None:
            return list(cached)

        radius = self._fov_radius(z)
        visible = []
        for idx, (ox, oy) in enumerate(self.objects):
            dist = ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
            if dist <= radius and not self._is_occluded(x, y, z, ox, oy):
                visible.append((idx, dist))
        cached_visible = tuple(visible)
        self._visible_cache[key] = cached_visible
        return list(cached_visible)

    def _valid_pos(self, x: int, y: int, z: int) -> bool:
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        if z < 1 or z > MAX_Z:
            return False
        if z in self.passable and not self.passable[z][x, y]:
            return False
        return True

    def step(self, include_details: bool = True) -> dict:
        self.step_count += 1
        prev_seen = len(self.seen_cells)
        prev_discovered = len(self.discovered)

        # Evaluate waypoint options
        waypoints = self._get_waypoints()
        efe_per_wp = {}
        G = np.zeros(len(waypoints))
        for i, wp in enumerate(waypoints):
            efe = self._evaluate_waypoint(wp)
            efe_per_wp[wp['name']] = efe
            G[i] = efe['total']

        gamma = 1.0 / max(self.agent.beta, 1e-8)
        probs = softmax(gamma * G)

        # Waypoint commitment: keep current unless reached, invalid, or clearly dominated
        need_new = True
        if self.current_wp is not None:
            tx, ty, tz = self.current_wp['target']
            x, y, z = self.drone_pos
            arrived = (x == tx and y == ty and z == tz)

            # For explore waypoints, check if target area is still unseen
            if self.current_wp['type'] == WP_EXPLORE:
                fov = self._fov_cells(tx, ty, tz)
                still_useful = len(fov - self.seen_cells) > 0
            else:
                still_useful = self.current_wp['name'] in efe_per_wp

            if still_useful and not arrived:
                # Re-evaluate current waypoint with fresh data
                cur_efe = self._evaluate_waypoint(self.current_wp)['total']
                best_efe = float(G.max())
                if best_efe - cur_efe < 1.5:
                    need_new = False  # stay committed

        if need_new:
            chosen_idx = int(np.random.choice(len(waypoints), p=probs))
            self.current_wp = waypoints[chosen_idx]
            self.wp_steps = 0

        chosen_wp = self.current_wp
        self.wp_steps += 1

        # Move one cell toward chosen waypoint
        prev_heading = self.prev_heading
        move = self._best_move_toward(chosen_wp)
        move_class = self._move_class(move, prev_heading) if move >= 0 else None
        if move >= 0:
            dx, dy, dz = ACTION_DELTAS[move]
            self.drone_pos = [self.drone_pos[0] + dx, self.drone_pos[1] + dy, self.drone_pos[2] + dz]
            cost = BATTERY_COST_VERT if dz != 0 else BATTERY_COST_MOVE
            self.battery = max(0, self.battery - cost)
            next_heading = self._horizontal_heading(move)
            if next_heading is not None:
                self.prev_heading = next_heading

        # Update exploration state and discover objects in FOV
        x, y, z = self.drone_pos
        self.seen_cells |= self._fov_cells(x, y, z)
        for obj_idx, _dist in self._visible_objects(x, y, z):
            self.discovered.add(obj_idx)

        observations = self._generate_observations()
        self._bayesian_update(observations)
        self._update_learning(chosen_wp, move_class, prev_seen, prev_discovered)

        # Auto-confirm when drone arrives at confirm waypoint destination
        reward = 0.0
        if (chosen_wp['type'] == WP_CONFIRM
                and not self.found_target and not self.mission_failed):
            idx = chosen_wp['idx']
            ox, oy = self.objects[idx]
            if x == ox and y == oy and z == 1:
                if idx == self.true_target:
                    reward = 10.0
                    self.found_target = True
                    self.confirmed_object = idx
                else:
                    reward = -8.0
                    self.mission_failed = True
                    self.confirmed_object = idx
        self.total_reward += reward
        self.flight_path.append(list(self.drone_pos))

        if not include_details:
            return {
                'step': self.step_count,
                'reward': reward,
                'total_reward': self.total_reward,
                'found_target': self.found_target,
                'mission_failed': self.mission_failed,
                'battery': self.battery,
                'n_discovered': len(self.discovered),
            }

        wp_probs = {waypoints[i]['name']: float(probs[i]) for i in range(len(waypoints))}
        move_name = ACTION_NAMES[move] if move >= 0 else 'stay'

        return {
            'step': self.step_count,
            'position': list(self.drone_pos),
            'action': move_name,
            'waypoint': chosen_wp['name'],
            'waypoint_type': chosen_wp['type'],
            'reward': reward,
            'total_reward': self.total_reward,
            'found_target': self.found_target,
            'mission_failed': self.mission_failed,
            'battery': self.battery,
            'confirmed_object': OBJECT_NAMES[self.confirmed_object] if self.confirmed_object >= 0 else None,
            'beliefs': {
                'target_belief': [
                    {'name': OBJECT_NAMES[i],
                     'position': list(self.objects[i]) if i in self.discovered else None,
                     'prob': round(float(self.target_belief[i]), 4),
                     'discovered': i in self.discovered}
                    for i in range(self.n_objects)
                ],
                'disc_quality': [
                    {'altitude': z,
                     'believed_accuracy': round(float(dirichlet_expected(self.disc_conc[z])[0]), 3)}
                    for z in range(1, MAX_Z + 1)
                ],
                'search_profile': [
                    {'altitude': z,
                     'search_gain': round(self._learned_search_efficiency(z), 3),
                     'clearance': round(self._learned_blocker_clearance(z), 3)}
                    for z in EXPLORE_ALTITUDES
                ],
                'direction_profile': self._direction_profile(),
            },
            'observations': self.obs_log[-5:],
            'efe': efe_per_wp,
            'policy_probs': wp_probs,
            'scan_state': self._scan_state(),
        }

    def _update_learning(self, waypoint: dict, move_class: str | None, prev_seen: int, prev_discovered: int) -> None:
        x, y, z = self.drone_pos
        clearance = self._fov_clearance(x, y, z)
        self.blocker_conc[z, 0] += clearance
        self.blocker_conc[z, 1] += 1.0 - clearance

        if waypoint['type'] != WP_EXPLORE:
            return

        new_cells = len(self.seen_cells) - prev_seen
        new_objects = len(self.discovered) - prev_discovered
        gain = new_objects * 2.5 + new_cells / max(len(self._fov_cells(x, y, z)), 1)
        self.search_gain_sum[z] += gain
        self.search_gain_count[z] += 1.0
        if move_class is not None:
            idx = DIRECTION_INDEX[move_class]
            self.dir_gain_sum[z, idx] += gain
            self.dir_gain_count[z, idx] += 1.0

    # ─── Observation generation (true model) ───

    def _generate_observations(self) -> list[tuple[int, int, int]]:
        """Generate observations for visible objects. Returns [(obj_idx, obs_type, altitude)]."""
        x, y, z = self.drone_pos
        visible = self._visible_objects(x, y, z)
        obs = []

        for obj_idx, dist in visible:
            quality = TRUE_QUALITY[z]
            is_target = (obj_idx == self.true_target)

            if quality < DISC_THRESHOLD:
                obs.append((obj_idx, OBS_BLOB, z))
                self.obs_log.append({'object': OBJECT_NAMES[obj_idx], 'altitude': z, 'obs': 'blob'})
            else:
                correct = np.random.random() < quality
                if is_target:
                    obs_type = OBS_TARGET_SIG if correct else OBS_DISTRACTOR_SIG
                else:
                    obs_type = OBS_DISTRACTOR_SIG if correct else OBS_TARGET_SIG
                label = 'target-like' if obs_type == OBS_TARGET_SIG else 'distractor-like'
                obs.append((obj_idx, obs_type, z))
                self.obs_log.append({'object': OBJECT_NAMES[obj_idx], 'altitude': z, 'obs': label})
                self.inspected.add(obj_idx)
                self.info_altitude_sum += float(z)
                self.info_altitude_count += 1.0

        return obs

    # ─── Bayesian inference (agent's model) ───

    def _bayesian_update(self, observations: list[tuple[int, int, int]]):
        """Update target beliefs and Dirichlet concentrations."""
        for obj_idx, obs_type, alt_z in observations:
            if obs_type == OBS_BLOB:
                continue

            believed_acc = dirichlet_expected(self.disc_conc[alt_z])[0]
            believed_err = 1.0 - believed_acc

            likelihood = np.ones(self.n_objects)
            for h in range(self.n_objects):
                if h == obj_idx:
                    likelihood[h] = believed_acc if obs_type == OBS_TARGET_SIG else believed_err
                else:
                    likelihood[h] = believed_err if obs_type == OBS_TARGET_SIG else believed_acc

            posterior = self.target_belief * likelihood
            t = posterior.sum()
            if t > 1e-16:
                self.target_belief = posterior / t

            # Dirichlet update: weight by P(observation was correct)
            if obs_type == OBS_TARGET_SIG:
                p_correct = self.target_belief[obj_idx]
            else:
                p_correct = 1.0 - self.target_belief[obj_idx]
            self.disc_conc[alt_z, 0] += p_correct * 0.3
            self.disc_conc[alt_z, 1] += (1.0 - p_correct) * 0.3

    # ─── Waypoint evaluation ───

    def _get_waypoints(self) -> list[dict]:
        """Available waypoint actions based on current discovery state."""
        wps = []
        for idx in sorted(self.discovered):
            ox, oy = self.objects[idx]
            wps.append({'type': WP_SCAN, 'idx': idx,
                        'name': f'Scan {OBJECT_NAMES[idx]}', 'target': (ox, oy, 2)})
            wps.append({'type': WP_CONFIRM, 'idx': idx,
                        'name': f'Confirm {OBJECT_NAMES[idx]}', 'target': (ox, oy, 1)})
        # Dynamic explore waypoints from unseen frontier
        wps.extend(self._explore_waypoints())
        return wps

    def _explore_waypoints(self) -> list[dict]:
        """Generate frontier waypoints from unseen cells using learned search priors."""
        all_cells = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        unseen = all_cells - self.seen_cells
        if not unseen:
            return []

        # Score unseen cells by expected frontier gain, learned search efficiency,
        # and learned blocker clearance. Sample a subset to keep it fast.
        candidates = list(unseen)
        candidate_cap = min(len(candidates), max(30, self.grid_size * 4))
        if len(candidates) > candidate_cap:
            np.random.shuffle(candidates)
            candidates = candidates[:candidate_cap]

        scored = []
        for cx, cy in candidates:
            for explore_z in EXPLORE_ALTITUDES:
                if not self._valid_pos(cx, cy, explore_z):
                    continue
                fov = self._fov_cells(cx, cy, explore_z)
                new_count = len(fov - self.seen_cells)
                if new_count <= 0:
                    continue
                frontier_gain = new_count / max(len(fov), 1)
                local_clearance = self._fov_clearance(cx, cy, explore_z)
                learned_gain = self._learned_search_efficiency(explore_z)
                learned_clearance = self._learned_blocker_clearance(explore_z)
                score = frontier_gain * (0.5 + learned_gain) * (0.5 + learned_clearance) * (0.5 + local_clearance)
                scored.append((score, new_count, explore_z, cx, cy, local_clearance, learned_gain, learned_clearance))

        scored.sort(reverse=True)

        # Pick top candidates, ensuring spatial diversity (min distance apart)
        chosen = []
        for score, count, explore_z, cx, cy, local_clearance, learned_gain, learned_clearance in scored:
            too_close = False
            for _, _, _, px, py, _, _, _ in chosen:
                if abs(cx - px) + abs(cy - py) < 4:
                    too_close = True
                    break
            if not too_close:
                chosen.append((score, count, explore_z, cx, cy, local_clearance, learned_gain, learned_clearance))
            if len(chosen) >= N_EXPLORE_WPS:
                break

        wps = []
        for i, (score, count, explore_z, cx, cy, local_clearance, learned_gain, learned_clearance) in enumerate(chosen):
            wps.append({
                'type': WP_EXPLORE,
                'idx': i,
                'name': f'Explore ({cx},{cy},z{explore_z})',
                'target': (cx, cy, explore_z),
                'new_cells': count,
                'score': score,
                'local_clearance': local_clearance,
                'learned_gain': learned_gain,
                'learned_clearance': learned_clearance,
            })
        return wps

    def _evaluate_waypoint(self, wp: dict) -> dict:
        """Compute EFE components for reaching a waypoint."""
        tx, ty, tz = wp['target']
        x, y, z = self.drone_pos
        dist = abs(x - tx) + abs(y - ty) + abs(z - tz)

        battery_frac = self.battery / self.battery_max if self.battery_max > 0 else 0.0
        urgency = 1.0 + 2.0 * (1.0 - battery_frac)
        explore_scale = max(0.5, battery_frac)
        dist_penalty = dist * (0.05 + 0.15 * (1.0 - battery_frac))

        ext = 0.0
        sal = 0.0
        nov = 0.0

        if wp['type'] == WP_CONFIRM:
            idx = wp['idx']
            p = float(self.target_belief[idx])
            ext = p * 10.0 - (1.0 - p) * 8.0
            if p < 0.55:
                ext = -10.0
            total = self.agent.w_extrinsic * ext * urgency - dist_penalty
            return {'extrinsic': float(ext), 'salience': 0.0, 'novelty': 0.0,
                    'total': float(total), 'distance': dist}

        if wp['type'] == WP_SCAN:
            idx = wp['idx']
            sal = self._single_obj_info_gain(idx, tz)
            if idx not in self.inspected:
                sal += 0.5
            sal *= 1.5
            believed_acc = dirichlet_expected(self.disc_conc[tz])[0]
            if believed_acc >= DISC_THRESHOLD:
                nov = self._disc_param_info_gain(tz)
            ext = float(self.target_belief[idx]) * 2.0

        elif wp['type'] == WP_EXPLORE:
            new_fov = self._fov_cells(tx, ty, tz)
            new_cells = wp.get('new_cells', len(new_fov - self.seen_cells))
            total_fov = max(len(new_fov), 1)
            undiscovered_frac = max(0, self.n_objects - len(self.discovered)) / self.n_objects
            frontier_gain = new_cells / total_fov
            learned_gain = wp.get('learned_gain', self._learned_search_efficiency(tz))
            learned_clearance = wp.get('learned_clearance', self._learned_blocker_clearance(tz))
            local_clearance = wp.get('local_clearance', self._fov_clearance(tx, ty, tz))
            sal = frontier_gain * undiscovered_frac * (3.0 + learned_gain)
            sal *= 0.5 + 0.5 * local_clearance
            nov = self._disc_param_info_gain(tz) * (0.15 + 0.35 * learned_clearance)

        total = (self.agent.w_extrinsic * ext * urgency +
                 self.agent.w_salience * sal * explore_scale +
                 self.agent.w_novelty * nov * explore_scale -
                 dist_penalty)

        return {
            'extrinsic': float(ext),
            'salience': float(sal),
            'novelty': float(nov),
            'total': float(total),
            'distance': dist,
        }

    def _best_move_toward(self, waypoint: dict) -> int:
        """Return action index for the best single-cell move toward a waypoint."""
        tx, ty, tz = waypoint['target']
        x, y, z = self.drone_pos
        if x == tx and y == ty and z == tz:
            return -1

        current_dist = abs(x - tx) + abs(y - ty) + abs(z - tz)
        options = []
        for a in range(6):  # N, S, E, W, Up, Down
            dx, dy, dz = ACTION_DELTAS[a]
            nx, ny, nz = x + dx, y + dy, z + dz
            if self._valid_pos(nx, ny, nz):
                next_dist = abs(nx - tx) + abs(ny - ty) + abs(nz - tz)
                dist_gain = current_dist - next_dist
                move_class = self._move_class(a, self.prev_heading)
                clearance = self._fov_clearance(nx, ny, nz)
                score = 3.0 * dist_gain
                dir_bias = 0.0

                if waypoint['type'] == WP_EXPLORE:
                    next_fov = self._fov_cells(nx, ny, nz)
                    frontier_gain = len(next_fov - self.seen_cells) / max(len(next_fov), 1)
                    score += frontier_gain * (2.0 + self._learned_search_efficiency(nz))
                    score += 0.35 * clearance + 0.25 * self._learned_blocker_clearance(nz)
                    dir_bias = self._learned_direction_bias(nz, move_class)
                else:
                    if move_class in ('seek', 'forward'):
                        score += 0.2
                    if move_class == 'reverse':
                        score -= 0.3
                    score += 0.2 * clearance

                options.append((score, dir_bias, next_dist, a))

        if not options:
            return -1  # stuck

        if waypoint['type'] == WP_EXPLORE:
            best_base = max(item[0] for item in options)
            rescored = []
            for base_score, dir_bias, next_dist, action in options:
                final_score = base_score
                if best_base - base_score < 0.35:
                    final_score += 0.5 * dir_bias
                rescored.append((final_score, next_dist, action))
            rescored.sort(key=lambda item: (-item[0], item[1], item[2]))
            return int(rescored[0][2])

        options.sort(key=lambda item: (-item[0], item[2], item[3]))
        return int(options[0][3])

    def _expected_obs_info_gain(self, z: int, visible: list[tuple[int, float]]) -> float:
        """Mutual information I(target; observation | altitude z, visible objects)."""
        believed_acc = dirichlet_expected(self.disc_conc[z])[0]
        if believed_acc < DISC_THRESHOLD:
            return 0.0

        current_H = self._belief_entropy()
        total_gain = 0.0

        for obj_idx, dist in visible:
            for obs_type in [OBS_TARGET_SIG, OBS_DISTRACTOR_SIG]:
                p_obs = 0.0
                for h in range(self.n_objects):
                    if h == obj_idx:
                        p_obs += self.target_belief[h] * (believed_acc if obs_type == OBS_TARGET_SIG else (1 - believed_acc))
                    else:
                        p_obs += self.target_belief[h] * ((1 - believed_acc) if obs_type == OBS_TARGET_SIG else believed_acc)
                if p_obs < 1e-16:
                    continue
                post = np.zeros(self.n_objects)
                for h in range(self.n_objects):
                    if h == obj_idx:
                        lik = believed_acc if obs_type == OBS_TARGET_SIG else (1 - believed_acc)
                    else:
                        lik = (1 - believed_acc) if obs_type == OBS_TARGET_SIG else believed_acc
                    post[h] = self.target_belief[h] * lik
                t = post.sum()
                if t > 1e-16:
                    post /= t
                post_H = float(-np.sum(post[post > 1e-16] * np.log(post[post > 1e-16])))
                total_gain += p_obs * (current_H - post_H)

        return max(total_gain, 0.0)

    def _single_obj_info_gain(self, obj_idx: int, z: int) -> float:
        """Expected info gain from observing single object at altitude z."""
        believed_acc = dirichlet_expected(self.disc_conc[z])[0]
        if believed_acc < DISC_THRESHOLD:
            return 0.0

        current_H = self._belief_entropy()
        gain = 0.0

        for obs_type in [OBS_TARGET_SIG, OBS_DISTRACTOR_SIG]:
            p_obs = 0.0
            for h in range(self.n_objects):
                if h == obj_idx:
                    p_obs += self.target_belief[h] * (believed_acc if obs_type == OBS_TARGET_SIG else (1 - believed_acc))
                else:
                    p_obs += self.target_belief[h] * ((1 - believed_acc) if obs_type == OBS_TARGET_SIG else believed_acc)
            if p_obs < 1e-16:
                continue
            post = np.zeros(self.n_objects)
            for h in range(self.n_objects):
                if h == obj_idx:
                    lik = believed_acc if obs_type == OBS_TARGET_SIG else (1 - believed_acc)
                else:
                    lik = (1 - believed_acc) if obs_type == OBS_TARGET_SIG else believed_acc
                post[h] = self.target_belief[h] * lik
            t = post.sum()
            if t > 1e-16:
                post /= t
            post_H = float(-np.sum(post[post > 1e-16] * np.log(post[post > 1e-16])))
            gain += p_obs * (current_H - post_H)

        return max(gain, 0.0)

    def _disc_param_info_gain(self, z: int) -> float:
        """Expected KL on Dirichlet from observing at altitude z."""
        conc = self.disc_conc[z]
        p = dirichlet_expected(conc)
        expected_kl = 0.0
        for outcome in range(2):
            post_conc = conc.copy()
            post_conc[outcome] += 0.3
            expected_kl += p[outcome] * kl_dirichlet(post_conc, conc)
        return expected_kl

    def _belief_entropy(self) -> float:
        b = self.target_belief[self.target_belief > 1e-16]
        if len(b) == 0:
            return 0.0
        return float(-np.sum(b * np.log(b)))

    def _scan_state(self) -> dict:
        x, y, z = self.drone_pos
        ended = self.found_target or self.mission_failed or self.battery <= 0
        return {
            'grid_size': self.grid_size,
            'max_z': MAX_Z,
            'buildings': [{'x': int(b['x']), 'y': int(b['y']), 'h': int(b['h'])}
                          for b in self.buildings],
            'objects': [
                {'name': OBJECT_NAMES[i],
                 'x': int(self.objects[i][0]) if i in self.discovered or ended else None,
                 'y': int(self.objects[i][1]) if i in self.discovered or ended else None,
                 'is_target': i == self.true_target and ended,
                 'inspected': i in self.inspected,
                 'discovered': i in self.discovered}
                for i in range(self.n_objects)
            ],
            'drone': list(self.drone_pos),
            'fov_radius': self._fov_radius(z),
            'n_objects': self.n_objects,
            'n_discovered': len(self.discovered),
            'explored_pct': round(len(self.seen_cells) / (self.grid_size * self.grid_size) * 100, 1),
            'target_found': self.found_target,
            'mission_failed': self.mission_failed,
            'battery': self.battery,
            'battery_max': self.battery_max,
            'episode': self.episode,
            'learned_search': [
                {'altitude': z,
                 'search_gain': round(self._learned_search_efficiency(z), 3),
                 'clearance': round(self._learned_blocker_clearance(z), 3)}
                for z in EXPLORE_ALTITUDES
            ],
            'learned_direction': self._direction_profile(),
        }

    def run_experiment(self, n_steps: int = 80) -> list[dict]:
        results = []
        for _ in range(n_steps):
            if self.found_target or self.mission_failed or self.battery <= 0:
                break
            results.append(self.step(include_details=True))
        return results

    def run_episode_summary(self, n_steps: int = 80) -> dict:
        steps = 0
        for _ in range(n_steps):
            if self.found_target or self.mission_failed or self.battery <= 0:
                break
            self.step(include_details=False)
            steps += 1

        return {
            'accuracy': 1.0 if self.found_target else 0.0,
            'reward': float(self.total_reward),
            'steps': float(steps),
            'success': 1.0 if self.found_target else 0.0,
            'failure': 1.0 if self.mission_failed else 0.0,
            'extras': {
                'avg_info_height': float(self.info_altitude_sum / max(self.info_altitude_count, 1.0)),
                'battery_out_rate': 1.0 if (self.battery <= 0 and not self.found_target and not self.mission_failed) else 0.0,
                'steps_to_success': float(self.step_count if self.found_target else 0.0),
            },
        }

    def get_config(self) -> dict:
        return {
            'scenario': 'drone_search',
            'scenario_name': 'Object Discrimination',
            'agents': list(AGENTS.keys()),
            'current_agent': self.agent.name,
            'grid_size': self.grid_size,
            'max_z': MAX_Z,
            'battery_max': self.battery_max,
            'step': self.step_count,
            'episode': self.episode,
        }
