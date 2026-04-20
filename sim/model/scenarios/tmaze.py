"""T-Maze scenario wrapper — adapts existing maze_simulation for multi-scenario interface."""
from scenarios import BaseScenario
from maze_simulation import MazeSimulation


class TMazeScenario(BaseScenario):
    """The paper's extended T-maze with cue arm and hidden contexts."""

    def __init__(self):
        self.sim = MazeSimulation()

    def reset(self, agent_type: str = 'combined') -> None:
        self.sim.reset(agent_type)

    def step(self) -> dict:
        return self.sim.step()

    def run_experiment(self, n_steps: int = 32) -> list[dict]:
        return self.sim.run_experiment(n_steps)

    def get_config(self) -> dict:
        cfg = self.sim.get_config()
        cfg['scenario'] = 'tmaze'
        cfg['scenario_name'] = 'T-Maze (Paper)'
        return cfg
