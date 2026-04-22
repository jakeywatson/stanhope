"""Stable-context T-maze (Fig 8B / Fig 12 of Schwartenbeck et al. 2019).

Same asymmetric safe/risky geometry as the paper's main T-maze, but the
hidden context is drawn once per experiment and held constant for all 32
trials. The agent accumulates a Dirichlet prior over contexts from cue
observations; once its prior concentrates, the cue's expected information
gain collapses and active inference stops visiting the cue.

Mechanically identical to TMazeScenario except ``stable_context=True`` on
the underlying MazeSimulation.
"""
from scenarios import BaseScenario
from maze_simulation import MazeSimulation


class TMazeStableScenario(BaseScenario):
    def __init__(self):
        self.sim = MazeSimulation(stable_context=True)

    def reset(self, agent_type: str = 'combined') -> None:
        self.sim.reset(agent_type)
        # Stable-context demo needs a tight β so the collapsing salience term
        # cleanly flips cue_then_best → left_direct once the d-vector is
        # confident. Under the softer β used in volatile mode, the agent
        # keeps visiting the cue occasionally even when its belief is highly
        # concentrated and the drop in Fig 8C becomes invisible.
        ag = self.sim.agent
        if ag.name in ('active_inference', 'combined'):
            ag.beta = 0.125

    def step(self) -> dict:
        return self.sim.step()

    def run_experiment(self, n_steps: int = 32) -> list[dict]:
        return self.sim.run_experiment(n_steps)

    def get_config(self) -> dict:
        cfg = self.sim.get_config()
        cfg['scenario'] = 'tmaze_stable'
        cfg['scenario_name'] = 'T-Maze (Stable Context)'
        return cfg
