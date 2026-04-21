"""Drone Search v2 — occlusion-aware target/decoy discrimination in unknown sites.

See sim/drone_v2_spec.md for the full design. Phase A implements:
  - Environment generator (randomised buildings + context-biased object placement)
  - Line-of-sight observation model
  - Per-cell class belief, per-object target belief (flat priors)
  - Action space: explore / scan / confirm / reject / declare_done
  - Extrinsic-only EFE (greedy policy)

Phases B/C add salience + novelty + transferable Dirichlet priors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

try:
    from agents import AGENTS, Agent
    from math_utils import kl_dirichlet
except ImportError:  # headless execution from repo root
    from model.agents import AGENTS, Agent  # type: ignore
    from model.math_utils import kl_dirichlet  # type: ignore


# ─── Constants ────────────────────────────────────────────────────────────────

GRID_SIZE = 15
MAX_Z = 4
ALTITUDES = (1, 2, 3, 4)

# Cell classes (index into class Dirichlet / per-cell belief).
CLS_EMPTY = 0
CLS_BUILDING = 1
CLS_DECOY = 2
CLS_TARGET = 3
N_CLASSES = 4
CLASS_NAMES = ['empty', 'building', 'decoy', 'target']

# Sensor: altitude-dependent base accuracy (fixed, known to agent).
# Noisy enough that single-obs commitment is a losing strategy — greedy's
# downfall, combined's opportunity.
SENSOR_ACCURACY = {1: 0.92, 2: 0.78, 3: 0.62, 4: 0.48}

# FOV radius (chebyshev): z=1 → self only, z=2 → 3x3, z=3 → 5x5, z=4 → 7x7.
def fov_radius(z: int) -> int:
    return max(0, z - 1)

# Environment generator tuning.
BUILDING_DENSITY_RANGE = (0.12, 0.22)           # fraction of cells with buildings
BUILDING_HEIGHT_WEIGHTS = [0.45, 0.30, 0.15, 0.10]  # P(h=1..4)
BUILDING_GROWTH_PROB = 0.55                     # prob of growing from a seed to neighbour
OBJECT_COUNT_RANGE = (3, 6)
P_OBJECT_BASE = 0.30                            # base object placement prob per eligible cell
P_OBJECT_BLDG_NEIGHBOUR_BOOST = 0.70            # adds up to 0.70 when all 4 neighbours are buildings
P_TARGET_BASE = 0.50                            # target fraction for each placed object
P_TARGET_OPENNESS_PENALTY = 0.06                # subtracts up to ~0.24 in very open cells
# Rough marginal rate of objects per empty cell (for declare_done prior estimates).
AVG_OBJECTS_PER_CELL = 0.035

# Battery.
BATTERY_MAX = 200
BATTERY_COST_MOVE = 1
BATTERY_COST_VERT = 1

# Reward. FP is harshest — committing to a decoy is worse than missing a target,
# which is what forces careful scanning before confirmation.
REWARD_TP = 1.0
REWARD_TN = 0.5
REWARD_FP = -4.0
REWARD_FN = -1.5

# Episode termination.
MAX_STEPS = 400

# Action waypoint types.
WP_EXPLORE = 'explore'
WP_SCAN = 'scan'
WP_CONFIRM = 'confirm'
WP_REJECT = 'reject'
WP_DECLARE = 'declare'

# Commitment hysteresis — current waypoint stays unless beaten by this margin.
COMMIT_MARGIN = 1.5

# EFE term scales (balance salience/novelty magnitudes against reward units).
SALIENCE_SCALE = 0.7
NOVELTY_SCALE = 1.5

# How much the learned world_alpha biases the initial cell-class belief at the
# start of each episode. 0 = pure uniform (no transfer), 1 = fully use α.
# A small value gives a meaningful learning curve without over-skepticism.
LEARNED_PRIOR_WEIGHT = 0.4

# Observation-model maths helpers.
def wrong_class_mass(z: int) -> float:
    """P(obs = a given wrong class | true_class) = (1-a(z))/3."""
    return (1.0 - SENSOR_ACCURACY[z]) / (N_CLASSES - 1)


def _bernoulli_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


# ─── State dataclasses ──────────────────────────────────────────────────────

@dataclass
class ObjectState:
    idx: int
    x: int
    y: int
    is_target: bool
    discovered: bool = False
    # per-object target belief P(target) — set once discovered
    target_belief: float = 0.5
    resolved: str | None = None    # None | 'confirmed' | 'rejected'


@dataclass
class Waypoint:
    wp_type: str
    target: tuple[int, int, int] | None  # (x, y, z) for explore/scan, None otherwise
    obj_idx: int | None = None
    name: str = ''


# ─── Main scenario class ────────────────────────────────────────────────────

class DroneSearchV2Scenario:
    """Phase A implementation: env + observation model + extrinsic-only planner."""

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.episode = 0
        self.agent = AGENTS['greedy']  # default for Phase A
        # Transferable Dirichlet prior over cell classes (persists across reset,
        # cleared by hard_reset). Phase B: single global bin. Phase C: per-ctx.
        self.world_alpha = np.ones(N_CLASSES, dtype=float)
        self.reset('greedy')

    # ─── Reset / env generation ────────────────────────────────────────────

    def reset(self, agent_type: str = 'greedy') -> None:
        """Start a new episode (fresh world, preserving any learned priors)."""
        self.agent = AGENTS.get(agent_type, AGENTS['greedy'])

        # Ground truth (hidden from drone).
        self._generate_buildings()
        self._generate_objects()

        # Drone state.
        self.drone = [0, 0, 2]   # start NW corner at altitude 2
        self.battery = BATTERY_MAX
        self.step_count = 0
        self.episode += 1

        # Beliefs (flat prior for Phase A).
        self._init_belief_map()
        self.object_beliefs: dict[int, ObjectState] = {}
        self.discovered_idxs: set[int] = set()
        self.resolved_idxs: set[int] = set()

        # Mission-outcome tracking.
        self.confirmations: list[int] = []   # obj_idx list
        self.rejections: list[int] = []
        self.declared_done = False
        self.terminal_score: float | None = None
        self.outcome_counts = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

        # Waypoint commitment.
        self.current_wp: Waypoint | None = None

        # Observation log (newest last).
        self.obs_log: list[dict] = []

    def hard_reset(self, agent_type: str = 'greedy') -> None:
        """Reset AND wipe learned priors (use between experiments)."""
        self.world_alpha = np.ones(N_CLASSES, dtype=float)
        self.reset(agent_type)

    def _generate_buildings(self) -> None:
        """Cluster-based generator: pick seeds, grow via neighbour expansion."""
        H = W = self.grid_size
        self.bldg_mask = np.zeros((H, W), dtype=bool)
        self.bldg_height = np.zeros((H, W), dtype=np.int32)

        density = np.random.uniform(*BUILDING_DENSITY_RANGE)
        target_count = int(density * H * W)

        # Seed with a handful of roots.
        n_seeds = max(2, target_count // 5)
        for _ in range(n_seeds):
            x, y = np.random.randint(0, H), np.random.randint(0, W)
            self.bldg_mask[x, y] = True

        # Grow outward until target_count reached.
        attempts = 0
        while self.bldg_mask.sum() < target_count and attempts < 400:
            attempts += 1
            xs, ys = np.where(self.bldg_mask)
            if len(xs) == 0:
                break
            i = np.random.randint(len(xs))
            x, y = int(xs[i]), int(ys[i])
            neighbours = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                          if 0 <= x + dx < H and 0 <= y + dy < W and not self.bldg_mask[x + dx, y + dy]]
            if not neighbours:
                continue
            if np.random.rand() < BUILDING_GROWTH_PROB:
                nx, ny = neighbours[np.random.randint(len(neighbours))]
                self.bldg_mask[nx, ny] = True

        # Assign heights.
        heights = np.random.choice([1, 2, 3, 4], size=self.bldg_mask.sum(),
                                    p=BUILDING_HEIGHT_WEIGHTS)
        self.bldg_height[self.bldg_mask] = heights

    def _generate_objects(self) -> None:
        """Place objects on non-building cells, biased toward building-adjacent
        spots. Each placed object is a target with prob that decreases with
        local openness."""
        H = W = self.grid_size
        # Per-cell probability of placing an object.
        p_obj = np.zeros((H, W), dtype=float)
        for x in range(H):
            for y in range(W):
                if self.bldg_mask[x, y]:
                    continue
                bn = self._count_neighbours(x, y, self.bldg_mask)
                p_obj[x, y] = P_OBJECT_BASE + P_OBJECT_BLDG_NEIGHBOUR_BOOST * (bn / 4.0)

        n_objects = np.random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1)
        candidates = [(x, y) for x in range(H) for y in range(W) if p_obj[x, y] > 0]
        if len(candidates) < n_objects:
            n_objects = len(candidates)
        weights = np.array([p_obj[x, y] for x, y in candidates])
        weights = weights / weights.sum()
        picks = np.random.choice(len(candidates), size=n_objects, replace=False, p=weights)

        self.objects: list[ObjectState] = []
        empty_mask = ~self.bldg_mask
        for obj_i, pick_idx in enumerate(picks):
            x, y = candidates[pick_idx]
            openness = self._count_neighbours(x, y, empty_mask)
            p_target = max(0.0, P_TARGET_BASE - P_TARGET_OPENNESS_PENALTY * openness)
            is_target = bool(np.random.rand() < p_target)
            self.objects.append(ObjectState(idx=obj_i, x=int(x), y=int(y),
                                            is_target=is_target))

    @staticmethod
    def _count_neighbours(x: int, y: int, mask: np.ndarray) -> int:
        H, W = mask.shape
        c = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and mask[nx, ny]:
                c += 1
        return c

    # ─── Belief map ─────────────────────────────────────────────────────────

    def _init_belief_map(self) -> None:
        """Flat 0.25 per-class belief. The learned world_alpha biases initial
        belief only via a mild blending (controlled by LEARNED_PRIOR_WEIGHT)
        so that a trained agent has slight prior expectations without becoming
        overly skeptical of new observations."""
        H = W = self.grid_size
        learned = self.world_alpha / float(self.world_alpha.sum())
        prior = (1.0 - LEARNED_PRIOR_WEIGHT) * np.full(N_CLASSES, 0.25) \
                + LEARNED_PRIOR_WEIGHT * learned
        self.belief = np.tile(prior.astype(float), (H, W, 1))
        # Seen-cells tracker — cells for which we have at least one observation.
        self.seen_mask = np.zeros((H, W), dtype=bool)

    def _current_class_prior(self, x: int, y: int) -> np.ndarray:
        """Prior over cell class given current context."""
        learned = self.world_alpha / float(self.world_alpha.sum())
        return (1.0 - LEARNED_PRIOR_WEIGHT) * np.full(N_CLASSES, 0.25) \
               + LEARNED_PRIOR_WEIGHT * learned

    # ─── Line-of-sight + observation ────────────────────────────────────────

    def _has_los(self, dx: int, dy: int, dz: int, tx: int, ty: int) -> bool:
        """Ray from drone (dx,dy,dz) → target cell (tx, ty, 0.5).
        Blocked iff any intermediate cell has estimated height ≥ ray altitude there.
        For Phase A, we use GROUND TRUTH heights for the observation realiser
        (the drone's own belief-based LOS prediction is used by the planner in
        Phase B/C)."""
        steps = max(abs(dx - tx), abs(dy - ty))
        if steps == 0:
            return True
        for s in range(1, steps):
            t = s / steps
            ix = int(round(dx + (tx - dx) * t))
            iy = int(round(dy + (ty - dy) * t))
            if (ix, iy) == (tx, ty) or (ix, iy) == (dx, dy):
                continue
            ray_alt = dz + (0.5 - dz) * t
            if self.bldg_height[ix, iy] >= ray_alt:
                return False
        return True

    def _fov_cells(self, x: int, y: int, z: int) -> list[tuple[int, int]]:
        r = fov_radius(z)
        H = W = self.grid_size
        out = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cx, cy = x + dx, y + dy
                if 0 <= cx < H and 0 <= cy < W:
                    out.append((cx, cy))
        return out

    def _observe_cell(self, cx: int, cy: int, z: int) -> int:
        """Sample noisy observation of cell's true class given accuracy a(z)."""
        if self.bldg_mask[cx, cy]:
            true_cls = CLS_BUILDING
        else:
            # Check if an object sits here.
            obj = self._object_at(cx, cy)
            if obj is None:
                true_cls = CLS_EMPTY
            elif obj.is_target:
                true_cls = CLS_TARGET
            else:
                true_cls = CLS_DECOY
        a = SENSOR_ACCURACY[z]
        wrong = wrong_class_mass(z)
        probs = np.full(N_CLASSES, wrong)
        probs[true_cls] = a
        return int(np.random.choice(N_CLASSES, p=probs))

    def _object_at(self, x: int, y: int) -> ObjectState | None:
        for obj in self.objects:
            if obj.x == x and obj.y == y and obj.resolved is None:
                return obj
        return None

    def _update_cell_belief(self, cx: int, cy: int, obs_cls: int, z: int) -> None:
        """Bayesian update of the 4-class belief given a noisy observation."""
        a = SENSOR_ACCURACY[z]
        wrong = wrong_class_mass(z)
        likelihood = np.full(N_CLASSES, wrong)
        likelihood[obs_cls] = a
        # Note: this updates the *per-cell* belief. When obs_cls lands on a class
        # that is not the true class, Bayes still shifts probability mass —
        # that's the point: observations are probabilistic evidence.
        b = self.belief[cx, cy] * likelihood
        s = b.sum()
        if s > 0:
            self.belief[cx, cy] = b / s
        self.seen_mask[cx, cy] = True

    def _update_object_target_belief(self, obj: ObjectState, obs_cls: int, z: int) -> None:
        """Separate Beta-style update on the object's target-vs-decoy belief.
        Only observations of class ∈ {target, decoy} are informative."""
        if obs_cls not in (CLS_TARGET, CLS_DECOY):
            return
        a = SENSOR_ACCURACY[z]
        wrong = wrong_class_mass(z)
        p = obj.target_belief
        if obs_cls == CLS_TARGET:
            lik_target = a
            lik_decoy = wrong
        else:
            lik_target = wrong
            lik_decoy = a
        num = lik_target * p
        den = num + lik_decoy * (1 - p)
        obj.target_belief = num / den if den > 0 else p

    def _process_observations_at(self, x: int, y: int, z: int) -> None:
        """Gather all LOS-clear cells in FOV and update beliefs for each."""
        a_sens = SENSOR_ACCURACY[z]
        for cx, cy in self._fov_cells(x, y, z):
            if not self._has_los(x, y, z, cx, cy):
                continue
            obs_cls = self._observe_cell(cx, cy, z)
            self._update_cell_belief(cx, cy, obs_cls, z)
            # Online Dirichlet update on transferable class prior (weighted by
            # sensor accuracy — noisier obs contribute less pseudo-count mass).
            self.world_alpha[obs_cls] += a_sens
            # Object-level update if an object lives here.
            obj = self._object_at(cx, cy)
            if obj is not None:
                if not obj.discovered:
                    # Discover when cell belief on target/decoy is large enough.
                    obj_mass = self.belief[cx, cy, CLS_TARGET] + self.belief[cx, cy, CLS_DECOY]
                    if obj_mass > 0.55:
                        obj.discovered = True
                        self.discovered_idxs.add(obj.idx)
                        # Initialize target belief from cell belief ratio.
                        t = self.belief[cx, cy, CLS_TARGET]
                        d = self.belief[cx, cy, CLS_DECOY]
                        obj.target_belief = t / (t + d) if (t + d) > 0 else 0.5
                if obj.discovered:
                    self._update_object_target_belief(obj, obs_cls, z)
            self.obs_log.append({'cell': (cx, cy), 'z': z, 'obs': CLASS_NAMES[obs_cls]})
        self.obs_log = self.obs_log[-30:]

    # ─── Waypoint generation ────────────────────────────────────────────────

    def _generate_waypoints(self) -> list[Waypoint]:
        wps: list[Waypoint] = []

        # Explore waypoints: pick frontier cells (cells adjacent to seen ones
        # but unseen, or the highest-entropy cells). Sample several at each z.
        H = W = self.grid_size
        entropy_map = -np.sum(self.belief * np.log(np.maximum(self.belief, 1e-12)), axis=2)
        # Mask out cells that are firmly resolved (entropy very low).
        candidate_cells = []
        for x in range(H):
            for y in range(W):
                if entropy_map[x, y] > 0.5:
                    candidate_cells.append((x, y, entropy_map[x, y]))
        candidate_cells.sort(key=lambda t: -t[2])
        for (x, y, _) in candidate_cells[:8]:
            for z in ALTITUDES:
                wps.append(Waypoint(WP_EXPLORE, (x, y, z),
                                    name=f'Explore({x},{y},z{z})'))

        # Scan/Confirm/Reject waypoints per discovered unresolved object.
        for obj in self.objects:
            if obj.resolved is not None:
                continue
            if not obj.discovered:
                continue
            for z in ALTITUDES:
                wps.append(Waypoint(WP_SCAN, (obj.x, obj.y, z), obj_idx=obj.idx,
                                    name=f'Scan({obj.idx},z{z})'))
            wps.append(Waypoint(WP_CONFIRM, None, obj_idx=obj.idx,
                                name=f'Confirm({obj.idx})'))
            wps.append(Waypoint(WP_REJECT, None, obj_idx=obj.idx,
                                name=f'Reject({obj.idx})'))

        # Always include declare_done as an option.
        wps.append(Waypoint(WP_DECLARE, None, name='DeclareDone'))
        return wps

    # ─── EFE components ─────────────────────────────────────────────────────

    def _efe_extrinsic(self, wp: Waypoint) -> float:
        if wp.wp_type == WP_CONFIRM:
            obj = self.objects[wp.obj_idx]
            p = obj.target_belief
            return p * REWARD_TP + (1 - p) * REWARD_FP
        if wp.wp_type == WP_REJECT:
            obj = self.objects[wp.obj_idx]
            p = obj.target_belief
            return (1 - p) * REWARD_TN + p * REWARD_FN
        if wp.wp_type == WP_DECLARE:
            return self._expected_terminal_score()
        return 0.0

    def _expected_terminal_score(self) -> float:
        """Expected score if we declared_done right now. Accounts for undiscovered
        targets via a rough estimate: expected object count in unseen region times
        expected target fraction."""
        score = 0.0
        # Resolved objects contribute their expected outcome.
        for obj in self.objects:
            if obj.resolved == 'confirmed':
                score += obj.target_belief * REWARD_TP + (1 - obj.target_belief) * REWARD_FP
            elif obj.resolved == 'rejected':
                score += (1 - obj.target_belief) * REWARD_TN + obj.target_belief * REWARD_FN

        # Objects discovered but not yet resolved — if we decl_done now, we miss
        # them (treated as FNs if real targets, FPs if decoys — conservative:
        # count as FN always, since an unresolved target is a real miss).
        for obj in self.objects:
            if obj.discovered and obj.resolved is None:
                score += obj.target_belief * REWARD_FN  # real target missed
                # we don't credit rejection for decoys we failed to act on

        # Undiscovered-object penalty: E[undiscovered targets] = unseen_cells ×
        # marginal-object-rate × target-rate, with a small pessimism factor so
        # the drone doesn't prematurely declare done over unexplored regions.
        unseen = int((~self.seen_mask).sum())
        expected_undisc_targets = 1.4 * unseen * AVG_OBJECTS_PER_CELL * P_TARGET_BASE
        score += expected_undisc_targets * REWARD_FN
        return float(score)

    def _efe_salience(self, wp: Waypoint) -> float:
        """Expected entropy reduction on in-episode beliefs (per-cell class and
        per-object target) from the observations an action would produce."""
        if wp.wp_type == WP_EXPLORE:
            x, y, z = wp.target
            a = SENSOR_ACCURACY[z]
            wrong = wrong_class_mass(z)
            total = 0.0
            for cx, cy in self._fov_cells(x, y, z):
                if not self._has_los(x, y, z, cx, cy):
                    continue
                total += self._cell_info_gain(self.belief[cx, cy], a, wrong)
            return SALIENCE_SCALE * total
        if wp.wp_type == WP_SCAN:
            obj = self.objects[wp.obj_idx]
            z = wp.target[2]
            a = SENSOR_ACCURACY[z]
            wrong = wrong_class_mass(z)
            # Object-level target/decoy discrimination.
            p = obj.target_belief
            H_cur = _bernoulli_entropy(p)
            P_T = p * a + (1 - p) * wrong
            P_D = p * wrong + (1 - p) * a
            if P_T > 1e-12 and P_D > 1e-12:
                p_after_T = (p * a) / P_T
                p_after_D = (p * wrong) / P_D
                H_after_obj = P_T * _bernoulli_entropy(p_after_T) + P_D * _bernoulli_entropy(p_after_D)
            else:
                H_after_obj = H_cur
            obj_gain = max(0.0, H_cur - H_after_obj)
            # Also pick up the per-cell info gain at the object's cell.
            cell_gain = self._cell_info_gain(self.belief[obj.x, obj.y], a, wrong)
            return SALIENCE_SCALE * (obj_gain + cell_gain)
        return 0.0

    def _cell_info_gain(self, p_c: np.ndarray, a: float, wrong: float) -> float:
        """H[p(c)] − E_o[H[p(c|o)]] under the noisy-4-class sensor likelihood."""
        H_cur = -(p_c * np.log(np.maximum(p_c, 1e-16))).sum()
        H_after = 0.0
        for o in range(N_CLASSES):
            lik = np.full(N_CLASSES, wrong)
            lik[o] = a
            joint = p_c * lik
            P_o = joint.sum()
            if P_o < 1e-12:
                continue
            post = joint / P_o
            H_o = -(post * np.log(np.maximum(post, 1e-16))).sum()
            H_after += P_o * H_o
        return max(0.0, float(H_cur - H_after))

    def _efe_novelty(self, wp: Waypoint) -> float:
        """Expected KL on the transferable Dirichlet class prior from the
        observations this action would produce. Uses a quadratic approximation
        valid for small pseudo-count increments: KL(α+δe_o || α) ≈ ½ δ²(1/α_o − 1/α_0)."""
        alpha = self.world_alpha
        a0 = alpha.sum()
        inv_alpha = 1.0 / alpha
        inv_a0 = 1.0 / a0

        if wp.wp_type == WP_EXPLORE:
            x, y, z = wp.target
            a = SENSOR_ACCURACY[z]
            wrong = wrong_class_mass(z)
            total = 0.0
            for cx, cy in self._fov_cells(x, y, z):
                if not self._has_los(x, y, z, cx, cy):
                    continue
                p_c = self.belief[cx, cy]
                p_obs = p_c * a + (1 - p_c) * wrong  # predictive obs distribution
                total += max(0.0, 0.5 * a * a * float((p_obs * inv_alpha).sum() - inv_a0))
            return NOVELTY_SCALE * total
        if wp.wp_type == WP_SCAN:
            obj = self.objects[wp.obj_idx]
            z = wp.target[2]
            a = SENSOR_ACCURACY[z]
            wrong = wrong_class_mass(z)
            p_c = self.belief[obj.x, obj.y]
            p_obs = p_c * a + (1 - p_c) * wrong
            contribution = 0.5 * a * a * float((p_obs * inv_alpha).sum() - inv_a0)
            return NOVELTY_SCALE * max(0.0, contribution)
        return 0.0

    def _evaluate(self, wp: Waypoint) -> dict:
        ext = self._efe_extrinsic(wp)
        sal = self._efe_salience(wp)
        nov = self._efe_novelty(wp)
        dist = self._waypoint_distance(wp)
        dist_penalty = 0.05 * dist
        total = (self.agent.w_extrinsic * ext
                 + self.agent.w_salience * sal
                 + self.agent.w_novelty * nov
                 - dist_penalty)
        return {'extrinsic': ext, 'salience': sal, 'novelty': nov,
                'total': total, 'distance': dist}

    def _waypoint_distance(self, wp: Waypoint) -> float:
        dx, dy, dz = self.drone
        if wp.target is not None:
            tx, ty, tz = wp.target
            return abs(dx - tx) + abs(dy - ty) + abs(dz - tz)
        if wp.obj_idx is not None:
            obj = self.objects[wp.obj_idx]
            return abs(dx - obj.x) + abs(dy - obj.y) + abs(dz - 1)
        return 0.0

    # ─── Step / actuation ───────────────────────────────────────────────────

    def _best_move_toward(self, wp: Waypoint) -> tuple[int, int, int] | None:
        """Greedy one-cell step toward a target position."""
        if wp.target is None and wp.obj_idx is None:
            return None
        if wp.target is not None:
            tx, ty, tz = wp.target
        else:
            obj = self.objects[wp.obj_idx]
            tx, ty, tz = obj.x, obj.y, 1
        dx, dy, dz = self.drone
        if (dx, dy, dz) == (tx, ty, tz):
            return None
        # Step priority: altitude first (get to desired z), then x, then y.
        if dz != tz:
            return (0, 0, 1 if tz > dz else -1)
        if dx != tx:
            return (1 if tx > dx else -1, 0, 0)
        if dy != ty:
            return (0, 1 if ty > dy else -1, 0)
        return None

    def _resolve_object(self, obj_idx: int, action: str) -> float:
        """Commit a confirmation or rejection. Returns realised reward."""
        obj = self.objects[obj_idx]
        obj.resolved = action
        self.resolved_idxs.add(obj_idx)
        if action == 'confirmed':
            self.confirmations.append(obj_idx)
            if obj.is_target:
                self.outcome_counts['tp'] += 1
                return REWARD_TP
            self.outcome_counts['fp'] += 1
            return REWARD_FP
        # rejected
        self.rejections.append(obj_idx)
        if obj.is_target:
            self.outcome_counts['fn'] += 1
            return REWARD_FN
        self.outcome_counts['tn'] += 1
        return REWARD_TN

    def _finalize_declare_done(self) -> float:
        """Score remaining (discovered but unresolved) and undiscovered targets."""
        self.declared_done = True
        residual = 0.0
        for obj in self.objects:
            if obj.resolved is None and obj.is_target:
                # Unresolved target = FN.
                self.outcome_counts['fn'] += 1
                residual += REWARD_FN
        self.terminal_score = sum([
            self.outcome_counts['tp'] * REWARD_TP,
            self.outcome_counts['tn'] * REWARD_TN,
            self.outcome_counts['fp'] * REWARD_FP,
            self.outcome_counts['fn'] * REWARD_FN,
        ])
        return residual

    def step(self, include_details: bool = True) -> dict:
        """One simulation tick: pick waypoint, step one cell, observe, update beliefs."""
        self.step_count += 1
        reward_this_step = 0.0

        # If already done, return a terminal dict.
        if self.declared_done or self.battery <= 0 or self.step_count > MAX_STEPS:
            if not self.declared_done:
                reward_this_step += self._finalize_declare_done()
            return self._result_dict(reward_this_step, include_details)

        wps = self._generate_waypoints()
        evals = {wp.name: self._evaluate(wp) for wp in wps}
        if self.agent.force_uniform:
            # Random baseline — pick uniformly across available waypoints.
            best_wp = wps[np.random.randint(len(wps))]
            self.current_wp = best_wp
        else:
            # Pick argmax total; with ties, prefer lower distance.
            best_wp = max(wps, key=lambda w: (evals[w.name]['total'], -evals[w.name]['distance']))
            # Commitment: if we already have a waypoint, keep unless beaten by margin.
            if self.current_wp is not None and self._wp_still_valid(self.current_wp):
                cur_total = evals.get(self.current_wp.name, {}).get('total', -1e9)
                if evals[best_wp.name]['total'] - cur_total < COMMIT_MARGIN:
                    best_wp = self.current_wp
            self.current_wp = best_wp

        # Execute.
        if best_wp.wp_type == WP_DECLARE:
            reward_this_step += self._finalize_declare_done()
        elif best_wp.wp_type == WP_CONFIRM:
            reward_this_step += self._resolve_object(best_wp.obj_idx, 'confirmed')
            self.current_wp = None
        elif best_wp.wp_type == WP_REJECT:
            reward_this_step += self._resolve_object(best_wp.obj_idx, 'rejected')
            self.current_wp = None
        elif best_wp.wp_type in (WP_EXPLORE, WP_SCAN):
            mv = self._best_move_toward(best_wp)
            if mv is not None:
                ddx, ddy, ddz = mv
                self.drone[0] += ddx
                self.drone[1] += ddy
                self.drone[2] += ddz
                cost = BATTERY_COST_VERT if ddz != 0 else BATTERY_COST_MOVE
                self.battery -= cost
            # Observe after movement.
            self._process_observations_at(self.drone[0], self.drone[1], self.drone[2])

        return self._result_dict(reward_this_step, include_details, evals)

    def _wp_still_valid(self, wp: Waypoint) -> bool:
        if wp.wp_type == WP_DECLARE:
            return True
        if wp.obj_idx is not None:
            obj = self.objects[wp.obj_idx]
            if obj.resolved is not None:
                return False
            if wp.wp_type in (WP_CONFIRM, WP_REJECT, WP_SCAN) and not obj.discovered:
                return False
        if wp.target is not None:
            tx, ty, tz = wp.target
            # If we've arrived, waypoint complete.
            if (self.drone[0], self.drone[1], self.drone[2]) == (tx, ty, tz):
                return False
        return True

    # ─── Result packaging ───────────────────────────────────────────────────

    def _result_dict(self, step_reward: float, include_details: bool,
                      evals: dict | None = None) -> dict:
        total_score = self.terminal_score if self.terminal_score is not None else self._provisional_score()
        out = {
            'step': self.step_count,
            'position': list(self.drone),
            'battery': self.battery,
            'reward': step_reward,
            'total_reward': total_score,
            'declared_done': self.declared_done,
            'terminal_score': self.terminal_score,
            'outcome_counts': dict(self.outcome_counts),
        }
        if not include_details:
            return out
        if evals is not None and self.current_wp is not None:
            wp = self.current_wp
            out['waypoint'] = wp.name
            out['waypoint_type'] = wp.wp_type
            efe_packet = {name: {k: float(v) for k, v in e.items()} for name, e in evals.items()}
            out['efe'] = efe_packet
            out['policy_probs'] = self._policy_probs(evals)
        out['discovered'] = len(self.discovered_idxs)
        out['n_objects'] = len(self.objects)
        out['n_resolved'] = len(self.resolved_idxs)
        out['scene'] = self._scene_payload()
        out['belief_summary'] = self._belief_summary()
        out['world_alpha'] = [float(v) for v in self.world_alpha]
        out['class_names'] = CLASS_NAMES
        return out

    def _policy_probs(self, evals: dict) -> dict[str, float]:
        if not evals:
            return {}
        names = list(evals.keys())
        totals = np.array([evals[n]['total'] for n in names], dtype=float)
        beta = max(float(self.agent.beta), 1e-3)
        # β here is a precision/temperature. Lower β → sharper.
        z = (totals - totals.max()) / beta
        w = np.exp(z)
        w = w / max(w.sum(), 1e-12)
        return {name: float(p) for name, p in zip(names, w)}

    def _scene_payload(self) -> dict:
        """Ground-truth scene info for the renderer — buildings, objects, drone FOV."""
        buildings = []
        xs, ys = np.where(self.bldg_mask)
        for x, y in zip(xs.tolist(), ys.tolist()):
            buildings.append({'x': int(x), 'y': int(y), 'h': int(self.bldg_height[x, y])})
        objects = []
        for obj in self.objects:
            objects.append({
                'idx': obj.idx,
                'x': obj.x,
                'y': obj.y,
                'is_target': obj.is_target,
                'discovered': obj.discovered,
                'resolved': obj.resolved,
                'target_belief': float(obj.target_belief),
            })
        seen_xs, seen_ys = np.where(self.seen_mask)
        seen_cells = [[int(x), int(y)] for x, y in zip(seen_xs.tolist(), seen_ys.tolist())]
        z = int(self.drone[2])
        return {
            'grid_size': self.grid_size,
            'fov_radius': fov_radius(z),
            'buildings': buildings,
            'objects': objects,
            'seen_cells': seen_cells,
            'battery_max': BATTERY_MAX,
            'episode': self.episode,
        }

    def _belief_summary(self) -> dict:
        """Flattened per-cell belief for the heatmap overlay."""
        # 15x15x4 → list of [x, y, [p0, p1, p2, p3]]; we ship the full grid once
        # per step (225 cells × 4 floats ≈ cheap).
        H = W = self.grid_size
        cells = []
        for x in range(H):
            for y in range(W):
                cells.append({
                    'x': x,
                    'y': y,
                    'p': [float(v) for v in self.belief[x, y]],
                })
        return {
            'cells': cells,
            'class_names': CLASS_NAMES,
        }

    def _provisional_score(self) -> float:
        return sum([
            self.outcome_counts['tp'] * REWARD_TP,
            self.outcome_counts['tn'] * REWARD_TN,
            self.outcome_counts['fp'] * REWARD_FP,
            self.outcome_counts['fn'] * REWARD_FN,
        ])

    # ─── Episode loops ──────────────────────────────────────────────────────

    def run_experiment(self, n_steps: int = MAX_STEPS) -> list[dict]:
        results = []
        for _ in range(n_steps):
            if self.declared_done or self.battery <= 0:
                break
            results.append(self.step(include_details=True))
        return results

    def run_episode_summary(self, n_steps: int = MAX_STEPS) -> dict:
        for _ in range(n_steps):
            if self.declared_done or self.battery <= 0:
                break
            self.step(include_details=False)
        if not self.declared_done:
            self._finalize_declare_done()
        score = self.terminal_score or 0.0
        n_true_targets = sum(1 for o in self.objects if o.is_target)
        success = 1.0 if (self.outcome_counts['tp'] == n_true_targets
                          and self.outcome_counts['fp'] == 0
                          and self.outcome_counts['fn'] == 0) else 0.0
        return {
            'accuracy': success,
            'reward': float(score),
            'steps': float(self.step_count),
            'success': success,
            'failure': 1.0 - success,
            'extras': {
                'tp': float(self.outcome_counts['tp']),
                'fp': float(self.outcome_counts['fp']),
                'tn': float(self.outcome_counts['tn']),
                'fn': float(self.outcome_counts['fn']),
                'n_true_targets': float(n_true_targets),
                'discovered': float(len(self.discovered_idxs)),
            },
        }

    def get_config(self) -> dict:
        return {
            'scenario': 'drone_search_v2',
            'scenario_name': 'Unknown Site Search',
            'agents': list(AGENTS.keys()),
            'current_agent': self.agent.name,
            'grid_size': self.grid_size,
            'max_z': MAX_Z,
            'battery_max': BATTERY_MAX,
            'step': self.step_count,
            'episode': self.episode,
        }
