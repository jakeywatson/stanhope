"""Scenario runner — singleton dispatched by the TypeScript bridge.

Manages the currently active scenario and exposes a flat API
that Pyodide can call via pyodide.runPython().
"""
import numpy as np

from scenarios.tmaze import TMazeScenario
from scenarios.grid_maze import GridMazeScenario
from scenarios.drone_search import DroneSearchScenario

SCENARIOS = {
    'tmaze': TMazeScenario,
    'grid_maze': GridMazeScenario,
    'drone_search': DroneSearchScenario,
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
        if self.current_name == 'tmaze':
            accuracy_label = 'Reward Rate'
            reward_label = 'Reward / Trial'

        # Snapshot original agent weights so the global AGENTS registry can be
        # restored after force_extrinsic_only mutations.
        from agents import AGENTS
        original_weights = {
            name: (a.w_salience, a.w_novelty) for name, a in AGENTS.items()
        }

        try:
            agents = []
            for agent_type in agent_types:
                accuracy_sum = 0.0
                reward_sum = 0.0
                steps_sum = 0.0
                success_sum = 0.0
                failure_sum = 0.0
                extra_sums = {metric['key']: 0.0 for metric in extra_metrics}

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

                agents.append({
                    'agent': agent_type,
                    'episodes': max(1, episodes_per_agent),
                    'accuracy_sum': accuracy_sum,
                    'reward_sum': reward_sum,
                    'steps_sum': steps_sum,
                    'success_sum': success_sum,
                    'failure_sum': failure_sum,
                    'extra_sums': extra_sums,
                })

            return {
                'scenario': self.current_name,
                'scenario_name': cfg.get('scenario_name', self.current_name),
                'accuracy_label': accuracy_label,
                'reward_label': reward_label,
                'step_cap': n_steps,
                'extra_metrics': extra_metrics,
                'agents': agents,
            }
        finally:
            for name, (sal, nov) in original_weights.items():
                if name in AGENTS:
                    AGENTS[name].w_salience = sal
                    AGENTS[name].w_novelty = nov
            np.random.set_state(next_rng_state)

    def _summarize_episode(self, results: list[dict]) -> dict:
        if self.current_name == 'tmaze':
            if not results:
                return {
                    'accuracy': 0.0,
                    'reward': 0.0,
                    'steps': 0.0,
                    'success': 0.0,
                    'failure': 0.0,
                    'extras': {'cue_visit_rate': 0.0},
                }
            rewards = [float(r.get('reward', 0.0)) for r in results]
            rewarded = sum(1.0 for r in rewards if r > 0.0)
            failed = sum(1.0 for r in rewards if r < 0.0)
            n_trials = len(rewards)
            cue_visits = sum(1.0 for r in results if any(step.get('location') == 'cue' for step in r.get('trajectory', [])))
            return {
                'accuracy': rewarded / max(n_trials, 1),
                'reward': sum(rewards) / max(n_trials, 1),
                'steps': float(n_trials),
                'success': rewarded / max(n_trials, 1),
                'failure': failed / max(n_trials, 1),
                'extras': {'cue_visit_rate': cue_visits / max(n_trials, 1)},
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
        if self.current_name == 'tmaze':
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
        return []

    def get_config(self) -> dict:
        return self.scenario.get_config()

    def list_scenarios(self) -> list[dict]:
        return [
            {'id': 'tmaze', 'name': 'T-Maze (Paper)'},
            {'id': 'grid_maze', 'name': 'Room Search'},
            {'id': 'drone_search', 'name': 'Object Discrimination'},
        ]


# Global instance
runner = ScenarioRunner()
