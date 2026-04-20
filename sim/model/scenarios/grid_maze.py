"""Grid room search scenario with proper active inference.

A spatial navigation problem with genuine hidden states, information sources,
and learnable parameters — structurally parallel to the T-maze but in a grid.

Layout: 7x7 grid with 4 corner rooms, a central hub, and 2 informant posts.
The target is hidden in one of the 4 corners. Informant posts give noisy
directional cues about which corner holds the target.

Active inference components:
  - Extrinsic: move toward rooms with high P(target here) for reward
  - Salience: visit informants to reduce uncertainty about which room (hidden state)
  - Novelty: visit informants to learn how reliable they are (Dirichlet on cue accuracy)

This creates clear behavioral separation between agent types:
  - Greedy: runs to corners based on prior, ignores informants
  - Active Inference: detours to informants to learn which corner, then goes there
  - Active Learning: visits BOTH informants to learn their reliability, then acts
  - Combined: balances all three

Room entry is a commitment: entering the wrong room ends the episode.
That makes epistemic actions matter instead of letting agents brute-force rooms.
"""
from collections import deque

import numpy as np
from numpy import ndarray
from math_utils import softmax, log_stable, entropy, dirichlet_expected, kl_dirichlet
from agents import AGENTS, Agent

# Grid layout constants
ROWS, COLS = 7, 7

# Special locations (row, col)
HUB = (3, 3)
INFORMANT_A = (1, 3)   # North informant
INFORMANT_B = (3, 1)   # West informant

ROOM_NE = (0, 6)
ROOM_NW = (0, 0)
ROOM_SE = (6, 6)
ROOM_SW = (6, 0)

ROOMS = [ROOM_NE, ROOM_NW, ROOM_SE, ROOM_SW]
ROOM_NAMES = ['NE', 'NW', 'SE', 'SW']
INFORMANTS = [INFORMANT_A, INFORMANT_B]
INFORMANT_NAMES = ['North Post', 'West Post']
N_ROOMS = 4
GOAL_INFORMANT = 'informant'
GOAL_ROOM = 'room'

# Actions
NORTH = 0; SOUTH = 1; EAST = 2; WEST = 3
N_ACTIONS = 4
ACTION_NAMES = ['north', 'south', 'east', 'west']
ACTION_DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

# Observations
OBS_NEUTRAL = 0
OBS_CUE_NE = 1
OBS_CUE_NW = 2
OBS_CUE_SE = 3
OBS_CUE_SW = 4
OBS_REWARD = 5
OBS_EMPTY = 6
N_OBS = 7

# True informant reliabilities (agent doesn't know these)
TRUE_RELIABILITY_A = 0.80
TRUE_RELIABILITY_B = 0.65
TRUE_RELIABILITIES = [TRUE_RELIABILITY_A, TRUE_RELIABILITY_B]
TARGET_REWARD = 10.0
WRONG_ROOM_PENALTY = -8.0

# Grid layout: 0 = passable, 1 = wall
GRID = np.array([
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
], dtype=int)

GOAL_SPECS = [
    {'type': GOAL_INFORMANT, 'idx': 0, 'name': 'North Informant', 'pos': INFORMANT_A},
    {'type': GOAL_INFORMANT, 'idx': 1, 'name': 'West Informant', 'pos': INFORMANT_B},
    {'type': GOAL_ROOM, 'idx': 0, 'name': 'Room NE', 'pos': ROOM_NE},
    {'type': GOAL_ROOM, 'idx': 1, 'name': 'Room NW', 'pos': ROOM_NW},
    {'type': GOAL_ROOM, 'idx': 2, 'name': 'Room SE', 'pos': ROOM_SE},
    {'type': GOAL_ROOM, 'idx': 3, 'name': 'Room SW', 'pos': ROOM_SW},
]


def _grid_passable(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS and GRID[r, c] == 0


def _goal_distance_map(goal: tuple[int, int]) -> ndarray:
    dist = np.full((ROWS, COLS), np.inf)
    queue = deque([goal])
    dist[goal[0], goal[1]] = 0.0

    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if not _grid_passable(nr, nc):
                continue
            if dist[nr, nc] <= dist[r, c] + 1.0:
                continue
            dist[nr, nc] = dist[r, c] + 1.0
            queue.append((nr, nc))

    return dist


GOAL_DISTANCE_MAPS = {spec['pos']: _goal_distance_map(spec['pos']) for spec in GOAL_SPECS}


class GridMazeScenario:
    """Active inference agent searching rooms with informant cues."""

    def __init__(self):
        self.agent: Agent = AGENTS['combined']
        self.step_count = 0
        self._init_state()

    def _init_state(self):
        self.agent_pos = HUB
        self.step_count = 0
        self.total_reward = 0.0
        self.found_target = False
        self.mission_failed = False
        self.failure_reason: str | None = None
        self.path: list[tuple[int, int]] = [HUB]
        self.target_room = int(np.random.randint(N_ROOMS))
        self.target_belief = np.ones(N_ROOMS) / N_ROOMS
        # Dirichlet concentrations for informant accuracy: (2, 2)
        # Per informant: [correct_conc, incorrect_conc]
        # Slight positive prior: agent suspects informants are somewhat reliable
        self.info_conc = np.array([[1.5, 1.0], [1.5, 1.0]])
        self.visited_rooms: set[int] = set()
        self.informant_visits = [0, 0]
        self.last_cues: list[int | None] = [None, None]
        self.current_goal: dict | None = None
        self.goal_steps = 0
        self.prev_heading: tuple[int, int] | None = None

    def reset(self, agent_type: str = 'combined') -> None:
        self.agent = AGENTS.get(agent_type, AGENTS['combined'])
        self._init_state()

    def step(self, include_details: bool = True) -> dict:
        self.step_count += 1

        efe_per_action = {}
        action_scores = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            efe = self._evaluate_action(a)
            efe_per_action[ACTION_NAMES[a]] = efe
            action_scores[a] = efe['total']

        goals = self._get_goals()
        goal_scores = {}
        goal_values = np.zeros(len(goals))
        for i, goal in enumerate(goals):
            efe = self._evaluate_goal(goal)
            goal_scores[goal['name']] = efe
            goal_values[i] = efe['total']

        need_new_goal = True
        if self.current_goal is not None:
            arrived = self.agent_pos == self.current_goal['pos']
            if not arrived:
                current_value = self._evaluate_goal(self.current_goal)['total']
                best_value = float(goal_values.max()) if len(goal_values) > 0 else current_value
                if best_value - current_value < 0.9:
                    need_new_goal = False

        gamma = 1.0 / max(self.agent.beta, 1e-8)
        if need_new_goal and len(goals) > 0:
            goal_probs = softmax(gamma * goal_values)
            chosen_goal_idx = int(np.random.choice(len(goals), p=goal_probs))
            self.current_goal = goals[chosen_goal_idx]
            self.goal_steps = 0

        chosen_goal = self.current_goal or goals[0]
        self.goal_steps += 1

        current_goal_dist = self._goal_distance(chosen_goal['pos'])
        committed_scores = action_scores.copy()
        for a in range(N_ACTIONS):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if not self._passable(nr, nc):
                committed_scores[a] = -1e9
                continue
            next_dist = self._goal_distance(chosen_goal['pos'], (nr, nc))
            if not np.isfinite(next_dist):
                committed_scores[a] = -1e9
                continue
            dist_gain = current_goal_dist - next_dist
            committed_scores[a] += 1.4 * dist_gain
            heading = (dr, dc)
            if dist_gain > 0 and heading == self.prev_heading:
                committed_scores[a] += 0.15
            elif self.prev_heading is not None and heading == (-self.prev_heading[0], -self.prev_heading[1]):
                committed_scores[a] -= 0.15

        probs = softmax(gamma * committed_scores)

        for a in range(N_ACTIONS):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if not self._passable(nr, nc):
                probs[a] = 0.0
        total = probs.sum()
        if total < 1e-16:
            probs = np.ones(N_ACTIONS) / N_ACTIONS
        else:
            probs /= total

        chosen = self._best_action_toward(chosen_goal, committed_scores)
        if chosen < 0:
            chosen = int(np.random.choice(N_ACTIONS, p=probs))

        dr, dc = ACTION_DELTAS[chosen]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if self._passable(nr, nc):
            self.agent_pos = (nr, nc)
            self.prev_heading = (dr, dc)

        obs = self._generate_observation()
        self._bayesian_update(obs)

        reward = 0.0
        if self.agent_pos in ROOMS:
            room_idx = ROOMS.index(self.agent_pos)
            self.visited_rooms.add(room_idx)
            if room_idx == self.target_room:
                reward = TARGET_REWARD
                self.found_target = True
            else:
                reward = WRONG_ROOM_PENALTY
                self.mission_failed = True
                self.failure_reason = 'wrong_room'
        self.total_reward += reward
        self.path.append(self.agent_pos)

        if not include_details:
            return {
                'step': self.step_count,
                'reward': reward,
                'total_reward': self.total_reward,
                'found_target': self.found_target,
            }

        return {
            'step': self.step_count,
            'position': list(self.agent_pos),
            'action': ACTION_NAMES[chosen],
            'goal': chosen_goal['name'],
            'goal_type': chosen_goal['type'],
            'observation': obs,
            'observation_name': self._observation_name(obs),
            'reward': reward,
            'total_reward': self.total_reward,
            'found_target': self.found_target,
            'mission_failed': self.mission_failed,
            'failure_reason': self.failure_reason,
            'outcome': 'success' if self.found_target else ('wrong_room' if self.mission_failed else 'searching'),
            'target_room': ROOM_NAMES[self.target_room] if (self.found_target or self.mission_failed) else None,
            'beliefs': {
                'target_belief': [
                    {'room': ROOM_NAMES[i], 'prob': round(float(self.target_belief[i]), 4),
                     'position': list(ROOMS[i])}
                    for i in range(N_ROOMS)
                ],
            },
            'informants': self._informant_state(reveal_truth=self.found_target),
            'efe': efe_per_action,
            'goal_efe': goal_scores,
            'policy_probs': {ACTION_NAMES[i]: float(probs[i]) for i in range(N_ACTIONS)},
            'maze': self._maze_state(),
        }

    def _passable(self, r: int, c: int) -> bool:
        return 0 <= r < ROWS and 0 <= c < COLS and GRID[r, c] == 0

    def _generate_observation(self) -> int:
        pos = self.agent_pos
        if pos in INFORMANTS:
            info_idx = INFORMANTS.index(pos)
            self.informant_visits[info_idx] += 1
            reliability = TRUE_RELIABILITY_A if info_idx == 0 else TRUE_RELIABILITY_B
            if np.random.random() < reliability:
                obs = OBS_CUE_NE + self.target_room
            else:
                wrong_rooms = [r for r in range(N_ROOMS) if r != self.target_room]
                obs = OBS_CUE_NE + int(np.random.choice(wrong_rooms))
            self.last_cues[info_idx] = obs - OBS_CUE_NE
            return obs
        if pos in ROOMS:
            room_idx = ROOMS.index(pos)
            if room_idx == self.target_room:
                return OBS_REWARD
            else:
                return OBS_EMPTY
        return OBS_NEUTRAL

    def _bayesian_update(self, obs: int):
        if obs == OBS_NEUTRAL:
            return
        if obs == OBS_REWARD:
            self.target_belief[:] = 0.0
            room_idx = ROOMS.index(self.agent_pos)
            self.target_belief[room_idx] = 1.0
            return
        if obs == OBS_EMPTY:
            room_idx = ROOMS.index(self.agent_pos)
            self.target_belief[room_idx] = 0.0
            t = self.target_belief.sum()
            if t > 1e-16:
                self.target_belief /= t
            return
        if OBS_CUE_NE <= obs <= OBS_CUE_SW:
            info_idx = INFORMANTS.index(self.agent_pos)
            cued_room = obs - OBS_CUE_NE
            conc = self.info_conc[info_idx]
            believed_acc = dirichlet_expected(conc)[0]  # P(correct cue)
            believed_err = 1.0 - believed_acc
            likelihood = np.zeros(N_ROOMS)
            for room_i in range(N_ROOMS):
                if room_i == cued_room:
                    likelihood[room_i] = believed_acc
                else:
                    likelihood[room_i] = believed_err / (N_ROOMS - 1)
            posterior = self.target_belief * likelihood
            t = posterior.sum()
            if t > 1e-16:
                self.target_belief = posterior / t
            # Dirichlet update: weight by P(cue was correct)
            p_correct = self.target_belief[cued_room]
            self.info_conc[info_idx, 0] += p_correct * 0.5
            self.info_conc[info_idx, 1] += (1.0 - p_correct) * 0.5

    def _evaluate_action(self, action: int) -> dict:
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if not self._passable(nr, nc):
            return {'extrinsic': -10.0, 'salience': 0.0, 'novelty': 0.0, 'total': -10.0}

        next_pos = (nr, nc)
        ext = 0.0
        sal = 0.0
        nov = 0.0

        # --- Extrinsic: expected reward ---
        if next_pos in ROOMS:
            room_idx = ROOMS.index(next_pos)
            p_target = float(self.target_belief[room_idx])
            ext = p_target * TARGET_REWARD + (1.0 - p_target) * WRONG_ROOM_PENALTY
        else:
            for i in range(N_ROOMS):
                if self.target_belief[i] > 0.01:
                    curr_dist = abs(self.agent_pos[0] - ROOMS[i][0]) + abs(self.agent_pos[1] - ROOMS[i][1])
                    next_dist = abs(nr - ROOMS[i][0]) + abs(nc - ROOMS[i][1])
                    if next_dist < curr_dist:
                        ext += self.target_belief[i] * 0.5

        # --- Salience: expected information gain about target location ---
        if next_pos in INFORMANTS:
            info_idx = INFORMANTS.index(next_pos)
            sal = self._expected_info_gain(info_idx)
        elif next_pos in ROOMS:
            sal = 0.0
        else:
            for info_idx, info_pos in enumerate(INFORMANTS):
                curr_dist = abs(self.agent_pos[0] - info_pos[0]) + abs(self.agent_pos[1] - info_pos[1])
                next_dist = abs(nr - info_pos[0]) + abs(nc - info_pos[1])
                if next_dist < curr_dist:
                    potential_gain = self._expected_info_gain(info_idx)
                    sal += potential_gain * 0.3 / max(next_dist, 1)

        # --- Novelty: expected parameter info gain about informant reliability ---
        if next_pos in INFORMANTS:
            info_idx = INFORMANTS.index(next_pos)
            nov = self._param_info_gain(info_idx)
        else:
            for info_idx, info_pos in enumerate(INFORMANTS):
                curr_dist = abs(self.agent_pos[0] - info_pos[0]) + abs(self.agent_pos[1] - info_pos[1])
                next_dist = abs(nr - info_pos[0]) + abs(nc - info_pos[1])
                if next_dist < curr_dist:
                    potential_nov = self._param_info_gain(info_idx)
                    nov += potential_nov * 0.2 / max(next_dist, 1)

        total = (self.agent.w_extrinsic * ext +
                 self.agent.w_salience * sal +
                 self.agent.w_novelty * nov)

        return {
            'extrinsic': float(ext),
            'salience': float(sal),
            'novelty': float(nov),
            'total': float(total),
        }

    def _get_goals(self) -> list[dict]:
        return [dict(spec) for spec in GOAL_SPECS]

    def _goal_distance(self, goal_pos: tuple[int, int], pos: tuple[int, int] | None = None) -> float:
        point = self.agent_pos if pos is None else pos
        return float(GOAL_DISTANCE_MAPS[goal_pos][point[0], point[1]])

    def _evaluate_goal(self, goal: dict) -> dict:
        dist = self._goal_distance(goal['pos'])
        ext = 0.0
        sal = 0.0
        nov = 0.0

        if goal['type'] == GOAL_INFORMANT:
            info_idx = goal['idx']
            if self.agent.w_salience <= 0.0 and self.agent.w_novelty <= 0.0:
                return {
                    'extrinsic': -4.0,
                    'salience': 0.0,
                    'novelty': 0.0,
                    'total': -4.0 - 0.35 * dist,
                    'distance': float(dist),
                }
            revisit_scale = 1.0 / (1.0 + 0.25 * self.informant_visits[info_idx])
            sal = self._expected_info_gain(info_idx) * revisit_scale
            nov = self._param_info_gain(info_idx) * revisit_scale
        else:
            room_idx = goal['idx']
            p_target = float(self.target_belief[room_idx])
            ext = p_target * TARGET_REWARD + (1.0 - p_target) * WRONG_ROOM_PENALTY
            if room_idx in self.visited_rooms:
                ext -= 2.0

        total = (self.agent.w_extrinsic * ext +
                 self.agent.w_salience * sal +
                 self.agent.w_novelty * nov -
                 0.35 * dist)

        return {
            'extrinsic': float(ext),
            'salience': float(sal),
            'novelty': float(nov),
            'total': float(total),
            'distance': float(dist),
        }

    def _best_action_toward(self, goal: dict, action_scores: ndarray) -> int:
        current_dist = self._goal_distance(goal['pos'])
        options = []
        valid_options = []

        for a in range(N_ACTIONS):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if not self._passable(nr, nc):
                continue
            valid_options.append((action_scores[a], a))
            next_dist = self._goal_distance(goal['pos'], (nr, nc))
            if not np.isfinite(next_dist):
                continue
            dist_gain = current_dist - next_dist
            score = action_scores[a] + 2.0 * dist_gain
            heading = (dr, dc)
            if dist_gain > 0 and heading == self.prev_heading:
                score += 0.2
            elif self.prev_heading is not None and heading == (-self.prev_heading[0], -self.prev_heading[1]):
                score -= 0.2
            options.append((score, next_dist, a))

        if options:
            options.sort(key=lambda item: (-item[0], item[1], item[2]))
            return int(options[0][2])
        if valid_options:
            valid_options.sort(key=lambda item: (-item[0], item[1]))
            return int(valid_options[0][1])
        return -1

    def _expected_info_gain(self, info_idx: int) -> float:
        """Proper mutual information: I(target; observation | informant)."""
        current_H = self._belief_entropy()
        conc = self.info_conc[info_idx]
        believed_acc = dirichlet_expected(conc)[0]  # P(correct cue)
        believed_err = 1.0 - believed_acc
        expected_H = 0.0
        for cued_room in range(N_ROOMS):
            p_cue = 0.0
            for target in range(N_ROOMS):
                if target == cued_room:
                    p_obs = believed_acc
                else:
                    p_obs = believed_err / (N_ROOMS - 1)
                p_cue += p_obs * self.target_belief[target]
            if p_cue < 1e-16:
                continue
            posterior = np.zeros(N_ROOMS)
            for target in range(N_ROOMS):
                if target == cued_room:
                    lik = believed_acc
                else:
                    lik = believed_err / (N_ROOMS - 1)
                posterior[target] = self.target_belief[target] * lik
            ptot = posterior.sum()
            if ptot > 1e-16:
                posterior /= ptot
            post_H = float(-np.sum(posterior[posterior > 1e-16] *
                                   np.log(posterior[posterior > 1e-16])))
            expected_H += p_cue * post_H
        return max(current_H - expected_H, 0.0)

    def _param_info_gain(self, info_idx: int) -> float:
        """Expected KL divergence on Dirichlet from observing this informant."""
        conc = self.info_conc[info_idx]
        p = dirichlet_expected(conc)
        expected_kl = 0.0
        for outcome in range(2):
            posterior_conc = conc.copy()
            posterior_conc[outcome] += 0.5
            expected_kl += p[outcome] * kl_dirichlet(posterior_conc, conc)
        return expected_kl
        return expected_kl

    def _belief_entropy(self) -> float:
        b = self.target_belief[self.target_belief > 1e-16]
        if len(b) == 0:
            return 0.0
        return float(-np.sum(b * np.log(b)))

    def _observation_name(self, obs: int) -> str:
        if obs == OBS_NEUTRAL:
            return 'neutral'
        if obs == OBS_REWARD:
            return 'target found'
        if obs == OBS_EMPTY:
            return 'empty room'
        if OBS_CUE_NE <= obs <= OBS_CUE_SW:
            return f'cue: {ROOM_NAMES[obs - OBS_CUE_NE]}'
        return 'unknown'

    def _informant_state(self, reveal_truth: bool = False) -> list[dict]:
        return [
            {
                'name': INFORMANT_NAMES[i],
                'position': list(INFORMANTS[i]),
                'visits': self.informant_visits[i],
                'last_cue': ROOM_NAMES[self.last_cues[i]] if self.last_cues[i] is not None else None,
                'believed_accuracy': float(dirichlet_expected(self.info_conc[i])[0]),
                'true_reliability': float(TRUE_RELIABILITIES[i]) if reveal_truth else None,
            }
            for i in range(len(INFORMANTS))
        ]

    def _maze_state(self, reveal_target: bool = False) -> dict:
        target_revealed = self.found_target or reveal_target
        return {
            'rows': ROWS,
            'cols': COLS,
            'grid': GRID.tolist(),
            'agent': list(self.agent_pos),
            'rooms': [{'name': ROOM_NAMES[i], 'pos': list(ROOMS[i]),
                        'is_target': i == self.target_room and target_revealed}
                       for i in range(N_ROOMS)],
            'informants': [{'name': INFORMANT_NAMES[i], 'pos': list(INFORMANTS[i]), 'visits': self.informant_visits[i]}
                           for i in range(len(INFORMANTS))],
            'target_found': self.found_target,
            'target_room': ROOM_NAMES[self.target_room] if target_revealed else None,
        }

    def run_experiment(self, n_steps: int = 80) -> list[dict]:
        results = []
        for _ in range(n_steps):
            if self.found_target or self.mission_failed:
                break
            results.append(self.step(include_details=True))

        if results:
            if self.found_target or self.mission_failed:
                results[-1]['informants'] = self._informant_state(reveal_truth=True)
                results[-1]['maze'] = self._maze_state(reveal_target=True)
            else:
                results[-1]['mission_failed'] = True
                results[-1]['failure_reason'] = 'step_cap'
                results[-1]['outcome'] = 'step_cap'
                results[-1]['target_room'] = ROOM_NAMES[self.target_room]
                results[-1]['informants'] = self._informant_state(reveal_truth=True)
                results[-1]['maze'] = self._maze_state(reveal_target=True)
        return results

    def run_episode_summary(self, n_steps: int = 80) -> dict:
        steps = 0
        for _ in range(n_steps):
            if self.found_target or self.mission_failed:
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
                'north_informant_visits': float(self.informant_visits[0]),
                'west_informant_visits': float(self.informant_visits[1]),
            },
        }

    def get_config(self) -> dict:
        return {
            'scenario': 'grid_maze',
            'scenario_name': 'Room Search',
            'agents': list(AGENTS.keys()),
            'current_agent': self.agent.name,
            'rows': ROWS,
            'cols': COLS,
            'step': self.step_count,
        }
