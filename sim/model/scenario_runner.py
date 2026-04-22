"""Scenario runner — singleton dispatched by the TypeScript bridge.

Manages the currently active scenario and exposes a flat API
that Pyodide can call via pyodide.runPython().
"""
import numpy as np

from scenarios.tmaze import TMazeScenario
from scenarios.tmaze_stable import TMazeStableScenario
from scenarios.tmaze_learning import TMazeLearningScenario
from scenarios.grid_maze import GridMazeScenario
from scenarios.drone_search import DroneSearchScenario
from scenarios.drone_search_v2 import DroneSearchV2Scenario

# T-maze-family scenarios share the same summary/extra-metric/trial-curve logic.
TMAZE_FAMILY = {'tmaze', 'tmaze_stable', 'tmaze_learning'}

SCENARIOS = {
    'tmaze': TMazeScenario,
    'tmaze_stable': TMazeStableScenario,
    'tmaze_learning': TMazeLearningScenario,
    'grid_maze': GridMazeScenario,
    'drone_search': DroneSearchScenario,
    'drone_search_v2': DroneSearchV2Scenario,
}


class ScenarioRunner:
    def __init__(self):
        self.current_name: str = 'tmaze'
        self.scenario = TMazeScenario()

    def switch(self, name: str) -> dict:
        """Switch to a different scenario. Returns its config."""
        cls = SCENARIOS.get(name)
        if cls is None:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
        self.current_name = name
        self.scenario = cls()
        return self.scenario.get_config()

    def reset(self, agent_type: str = 'combined') -> None:
        self.scenario.reset(agent_type)

    def hard_reset(self, agent_type: str = 'combined') -> None:
        """Full reset including any learned parameters."""
        if hasattr(self.scenario, 'hard_reset'):
            self.scenario.hard_reset(agent_type)
        else:
            self.scenario.reset(agent_type)

    def set_params(self, beta: float, w_ext: float, w_sal: float, w_nov: float) -> None:
        """Update the current agent's parameters from UI sliders."""
        # TMazeScenario wraps MazeSimulation — agent lives there
        agent = getattr(self.scenario, 'agent', None)
        if agent is None:
            sim = getattr(self.scenario, 'sim', None)
            if sim is not None:
                agent = getattr(sim, 'agent', None)
        if agent is not None:
            agent.beta = beta
            agent.w_extrinsic = w_ext
            agent.w_salience = w_sal
            agent.w_novelty = w_nov

    def set_model_params(self, reward_prob: float, context_prob: float, reward_mag: float) -> None:
        """Update generative-model parameters (T-maze only)."""
        sim = getattr(self.scenario, 'sim', None)
        if sim is not None:
            model = getattr(sim, 'model', None)
            if model is not None:
                model.update_params(reward_prob=reward_prob,
                                    context_prob=context_prob,
                                    reward_mag=reward_mag)

    def step(self) -> dict:
        return self.scenario.step()

    def run_experiment(self, n_steps: int = 32) -> list[dict]:
        return self.scenario.run_experiment(n_steps)

    def train_one_episode(self, agent_type: str, n_steps: int) -> dict:
        """Run a single fresh episode for training mode. Preserves any
        cross-episode learned priors (e.g. world_alpha on drone_search_v2).
        Returns the summary and current learned state."""
        self.scenario.reset(agent_type)
        if hasattr(self.scenario, 'run_episode_summary'):
            summary = self.scenario.run_episode_summary(n_steps)
        else:
            results = self.scenario.run_experiment(n_steps)
            summary = self._summarize_episode(results)
        world_alpha = [float(v) for v in getattr(self.scenario, 'world_alpha', [])]
        return {'summary': summary, 'world_alpha': world_alpha}

    def benchmark_batch(self, n_steps: int = 32, episodes_per_agent: int = 1, agent_types: list[str] | None = None, force_extrinsic_only: bool = False) -> dict:
        """Run a batch of fresh episodes for each agent and return aggregate metrics.

        When ``force_extrinsic_only`` is True, every agent's ``w_salience`` and
        ``w_novelty`` are zeroed before each episode. This isolates the
        contribution of the EFE curiosity terms versus pure goal-directed
        (extrinsic-only) planning, while leaving the rest of the agent stack
        (β, transition planner, learning) intact.
        """
        cfg = self.scenario.get_config()
        if agent_types is None:
            agent_types = cfg.get('agents', [])
        extra_metrics = self._extra_metric_specs()
        base_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        next_rng_state = np.random.get_state()

        accuracy_label = 'Success Rate'
        reward_label = 'Avg Reward'
        if self.current_name in TMAZE_FAMILY:
            accuracy_label = 'Big-Reward Rate'
            reward_label = 'Reward / Trial'

        # Snapshot original agent weights so the global AGENTS registry can be
        # restored after force_extrinsic_only mutations. We also snapshot beta
        # because the ablation sharpens it to greedy levels (see below).
        from agents import AGENTS
        original_weights = {
            name: (a.w_salience, a.w_novelty, a.beta) for name, a in AGENTS.items()
        }

        try:
            agents = []
            collect_trial_curves = self.current_name in TMAZE_FAMILY
            for agent_type in agent_types:
                accuracy_sum = 0.0
                reward_sum = 0.0
                steps_sum = 0.0
                success_sum = 0.0
                failure_sum = 0.0
                extra_sums = {metric['key']: 0.0 for metric in extra_metrics}
                # Per-trial-index sums across the episodes in this batch.
                # Aligned to n_steps — shorter episodes are padded with 0.
                trial_curve_sums: dict[str, list[float]] = (
                    {'cue_visit': [0.0] * n_steps,
                     'risky_visit': [0.0] * n_steps,
                     'big_hit': [0.0] * n_steps}
                    if collect_trial_curves else {}
                )

                for episode_idx in range(max(1, episodes_per_agent)):
                    # Seed each agent's episode identically within the batch so the
                    # leaderboard does not depend on agent iteration order.
                    np.random.seed((base_seed + episode_idx) % np.iinfo(np.int32).max)
                    if hasattr(self.scenario, 'hard_reset'):
                        self.scenario.hard_reset(agent_type)
                    else:
                        self.scenario.reset(agent_type)
                    if force_extrinsic_only:
                        active_agent = getattr(self.scenario, 'agent', None)
                        if active_agent is None:
                            active_sim = getattr(self.scenario, 'sim', None)
                            if active_sim is not None:
                                active_agent = getattr(active_sim, 'agent', None)
                        if active_agent is not None:
                            active_agent.w_salience = 0.0
                            active_agent.w_novelty = 0.0
                            # Also sharpen the policy. Without this, soft beta
                            # over flat EFE values produces near-random action
                            # selection that the waypoint dispatcher rescues,
                            # masking the loss of curiosity. Greedy uses 0.125.
                            if active_agent.name not in ('greedy', 'random'):
                                active_agent.beta = 0.125
                    if hasattr(self.scenario, 'run_episode_summary'):
                        summary = self.scenario.run_episode_summary(n_steps)
                    else:
                        results = self.scenario.run_experiment(n_steps)
                        summary = self._summarize_episode(results)
                    accuracy_sum += summary['accuracy']
                    reward_sum += summary['reward']
                    steps_sum += summary['steps']
                    success_sum += summary['success']
                    failure_sum += summary['failure']
                    for metric in extra_metrics:
                        extra_sums[metric['key']] += float(summary.get('extras', {}).get(metric['key'], 0.0))
                    if collect_trial_curves:
                        curves = summary.get('trial_curves', {}) or {}
                        for key in trial_curve_sums:
                            per_trial = curves.get(key, [])
                            for t, v in enumerate(per_trial):
                                if t < n_steps:
                                    trial_curve_sums[key][t] += float(v)

                agent_entry = {
                    'agent': agent_type,
                    'episodes': max(1, episodes_per_agent),
                    'accuracy_sum': accuracy_sum,
                    'reward_sum': reward_sum,
                    'steps_sum': steps_sum,
                    'success_sum': success_sum,
                    'failure_sum': failure_sum,
                    'extra_sums': extra_sums,
                }
                if collect_trial_curves:
                    agent_entry['trial_curve_sums'] = trial_curve_sums
                agents.append(agent_entry)

            trial_curve_specs: list[dict] = []
            if self.current_name in TMAZE_FAMILY:
                # Paper Fig 8C / Fig 6C — per-trial-index mean probability.
                # Each spec names a metric the dashboard renders as its own
                # chart. ``disabled_scenarios`` hides charts that are
                # mechanically zero for a given variant (e.g. cue visits in
                # the no-cue active-learning scenario).
                trial_curve_specs = [
                    {
                        'key': 'cue_visit',
                        'label': 'P(cue visit) by trial',
                        'short_label': 'Cue',
                        'disabled_scenarios': ['tmaze_learning'],
                    },
                    {
                        'key': 'risky_visit',
                        'label': 'P(risky visit) by trial',
                        'short_label': 'Risky',
                    },
                    {
                        'key': 'big_hit',
                        'label': 'P(big reward) by trial',
                        'short_label': 'Big',
                    },
                ]
                trial_curve_specs = [
                    spec for spec in trial_curve_specs
                    if self.current_name not in spec.get('disabled_scenarios', [])
                ]

            return {
                'scenario': self.current_name,
                'scenario_name': cfg.get('scenario_name', self.current_name),
                'accuracy_label': accuracy_label,
                'reward_label': reward_label,
                'step_cap': n_steps,
                'extra_metrics': extra_metrics,
                'trial_curves': trial_curve_specs,
                'agents': agents,
            }
        finally:
            for name, (sal, nov, beta) in original_weights.items():
                if name in AGENTS:
                    AGENTS[name].w_salience = sal
                    AGENTS[name].w_novelty = nov
                    AGENTS[name].beta = beta
            np.random.set_state(next_rng_state)

    def _summarize_episode(self, results: list[dict]) -> dict:
        if self.current_name in TMAZE_FAMILY:
            if not results:
                return {
                    'accuracy': 0.0,
                    'reward': 0.0,
                    'steps': 0.0,
                    'success': 0.0,
                    'failure': 0.0,
                    'extras': {'cue_visit_rate': 0.0},
                    'trial_curves': {'cue_visit': [], 'risky_visit': [], 'big_hit': []},
                }
            rewards = [float(r.get('reward', 0.0)) for r in results]
            n_trials = len(rewards)

            def final_obs(trial):
                traj = trial.get('trajectory', [])
                return traj[-1].get('observation', '') if traj else ''

            def visited(trial, loc):
                return 1 if any(step.get('location') == loc for step in trial.get('trajectory', [])) else 0

            big_hits = sum(1.0 for r in results if final_obs(r) == 'big')
            no_reward_hits = sum(1.0 for r in results if final_obs(r) == 'none')
            cue_visits = sum(visited(r, 'cue') for r in results)
            # Per-trial indicators — these drive the Fig 6C / Fig 8C-style
            # time-course charts in the experimenter dashboard.
            trial_curves = {
                'cue_visit': [visited(r, 'cue') for r in results],
                'risky_visit': [visited(r, 'right') for r in results],
                'big_hit': [1 if final_obs(r) == 'big' else 0 for r in results],
            }
            return {
                # "Success" = big-reward rate. The safe arm pays out every time
                # so counting any positive reward as success would collapse the
                # leaderboard. Paper-faithful scoring scores risky wins only.
                'accuracy': big_hits / max(n_trials, 1),
                'reward': sum(rewards) / max(n_trials, 1),
                'steps': float(n_trials),
                'success': big_hits / max(n_trials, 1),
                'failure': no_reward_hits / max(n_trials, 1),
                'extras': {'cue_visit_rate': cue_visits / max(n_trials, 1)},
                'trial_curves': trial_curves,
            }

        if not results:
            return {'accuracy': 0.0, 'reward': 0.0, 'steps': 0.0, 'success': 0.0, 'failure': 0.0, 'extras': {}}

        last = results[-1]
        found_target = bool(last.get('found_target', False))
        mission_failed = bool(last.get('mission_failed', False))
        reward = float(last.get('total_reward', sum(float(r.get('reward', 0.0)) for r in results)))
        extras = {}
        if self.current_name == 'grid_maze':
            informants = last.get('informants', [])
            extras = {
                'north_informant_visits': float(informants[0].get('visits', 0.0)) if len(informants) > 0 else 0.0,
                'west_informant_visits': float(informants[1].get('visits', 0.0)) if len(informants) > 1 else 0.0,
            }
        return {
            'accuracy': 1.0 if found_target else 0.0,
            'reward': reward,
            'steps': float(len(results)),
            'success': 1.0 if found_target else 0.0,
            'failure': 1.0 if mission_failed else 0.0,
            'extras': extras,
        }

    def _extra_metric_specs(self) -> list[dict]:
        if self.current_name in TMAZE_FAMILY:
            # In the no-cue learning variant the cue arm is disabled, so a
            # cue-visit-rate metric would just be a flat-zero column.
            if self.current_name == 'tmaze_learning':
                return []
            return [
                {
                    'key': 'cue_visit_rate',
                    'label': 'Cue Visit Rate',
                    'short_label': 'Cue',
                    'format': 'percent',
                    'decimals': 1,
                    'denominator': 'episodes',
                }
            ]
        if self.current_name == 'grid_maze':
            return [
                {
                    'key': 'north_informant_visits',
                    'label': 'North Visits / Episode',
                    'short_label': 'North',
                    'format': 'number',
                    'decimals': 1,
                    'denominator': 'episodes',
                },
                {
                    'key': 'west_informant_visits',
                    'label': 'West Visits / Episode',
                    'short_label': 'West',
                    'format': 'number',
                    'decimals': 1,
                    'denominator': 'episodes',
                },
            ]
        if self.current_name == 'drone_search':
            return [
                {
                    'key': 'avg_info_height',
                    'label': 'Informative Height',
                    'short_label': 'Height',
                    'format': 'number',
                    'decimals': 2,
                    'denominator': 'episodes',
                },
                {
                    'key': 'battery_out_rate',
                    'label': 'Battery Out Rate',
                    'short_label': 'Battery',
                    'format': 'percent',
                    'decimals': 1,
                    'denominator': 'episodes',
                },
                {
                    'key': 'steps_to_success',
                    'label': 'Steps To Success',
                    'short_label': 'Succ Steps',
                    'format': 'number',
                    'decimals': 1,
                    'denominator': 'success',
                },
            ]
        if self.current_name == 'drone_search_v2':
            return [
                {
                    'key': 'tp',
                    'label': 'True Positives',
                    'short_label': 'TP',
                    'format': 'number',
                    'decimals': 2,
                    'denominator': 'episodes',
                },
                {
                    'key': 'fp',
                    'label': 'False Positives',
                    'short_label': 'FP',
                    'format': 'number',
                    'decimals': 2,
                    'denominator': 'episodes',
                },
                {
                    'key': 'fn',
                    'label': 'Misses (FN)',
                    'short_label': 'FN',
                    'format': 'number',
                    'decimals': 2,
                    'denominator': 'episodes',
                },
                {
                    'key': 'n_true_targets',
                    'label': 'True Targets / Ep',
                    'short_label': 'Tgts',
                    'format': 'number',
                    'decimals': 2,
                    'denominator': 'episodes',
                },
            ]
        return []

    def get_config(self) -> dict:
        return self.scenario.get_config()

    def list_scenarios(self) -> list[dict]:
        return [
            {'id': 'tmaze', 'name': 'T-Maze (Volatile Context)'},
            {'id': 'tmaze_stable', 'name': 'T-Maze (Stable Context)'},
            {'id': 'tmaze_learning', 'name': 'T-Maze (Active Learning — no cue)'},
            {'id': 'grid_maze', 'name': 'Room Search'},
            {'id': 'drone_search', 'name': 'Object Discrimination'},
            {'id': 'drone_search_v2', 'name': 'Unknown Site Search'},
        ]


# Global instance
runner = ScenarioRunner()
