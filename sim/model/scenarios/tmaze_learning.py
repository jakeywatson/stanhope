"""Active-learning T-maze (Fig 3 / Fig 6 of Schwartenbeck et al. 2019).

Stripped-down variant with *no cue* — just a safe arm (deterministic small
reward) and a risky arm (stable reward probability, learnable via a
Dirichlet on the A-matrix). The cue-then-best policy is disabled, so the
only way to learn about the risky arm is to sample it directly. This is
the canonical active-learning (parameter exploration) setting.

Expected behaviour (Fig 6C):
    * Active Learning starts by sampling the risky arm heavily (high novelty
      bonus over the flat prior) and the risky-visit rate decays monotonically
      as the Dirichlet converges.
    * Random / greedy / active-inference show a flat response profile — there
      is no hidden-state uncertainty to resolve, so salience contributes
      nothing, and without novelty the risky arm looks worse than the safe
      arm under the structured prior.
"""
from scenarios import BaseScenario
from maze_simulation import MazeSimulation


class TMazeLearningScenario(BaseScenario):
    def __init__(self):
        # stable_context + no_cue gives the Fig 3/6 task: single hidden
        # reward probability to learn, no cue available.
        self.sim = MazeSimulation(stable_context=True, no_cue=True)

    def reset(self, agent_type: str = 'combined') -> None:
        self.sim.reset(agent_type)
        # Paper-Fig-6-scale novelty. Our one-step EFE underestimates the
        # value of a novel observation relative to the paper's multi-step
        # horizon, so we boost w_novelty for the parameter-exploration
        # agents. At w_novelty≈1-2 the extrinsic gap dominates and neither
        # AL nor Combined ever samples the risky arm; at w_novelty=20 the
        # iconic Fig-6C early-sampling spike reappears.
        ag = self.sim.agent
        if ag.name in ('active_learning', 'combined'):
            ag.w_novelty = 20.0

    def step(self) -> dict:
        return self.sim.step()

    def run_experiment(self, n_steps: int = 32) -> list[dict]:
        return self.sim.run_experiment(n_steps)

    def get_config(self) -> dict:
        cfg = self.sim.get_config()
        cfg['scenario'] = 'tmaze_learning'
        cfg['scenario_name'] = 'T-Maze (Active Learning — no cue)'
        return cfg
