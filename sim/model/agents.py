"""Agent implementations for the T-maze task.

Each agent type enables/disables different components of the Expected Free Energy:
- Greedy: extrinsic only
- Random: extrinsic only, very high β (imprecise)
- ActiveLearning: extrinsic + novelty (parameter exploration)
- ActiveInference: extrinsic + salience (hidden state exploration)
- Combined: all three components
"""
import numpy as np


class Agent:
    """Base configuration for an agent type."""

    def __init__(
        self,
        name: str,
        beta: float = 1.0,
        enable_extrinsic: bool = True,
        enable_salience: bool = False,
        enable_novelty: bool = False,
        w_extrinsic: float | None = None,
        w_salience: float | None = None,
        w_novelty: float | None = None,
        force_uniform: bool = False,
    ):
        self.name = name
        self.beta = beta
        self.w_extrinsic = w_extrinsic if w_extrinsic is not None else (1.0 if enable_extrinsic else 0.0)
        self.w_salience = w_salience if w_salience is not None else (1.0 if enable_salience else 0.0)
        self.w_novelty = w_novelty if w_novelty is not None else (1.0 if enable_novelty else 0.0)
        self.force_uniform = force_uniform

    @property
    def enable_extrinsic(self) -> bool:
        return self.w_extrinsic > 0

    @property
    def enable_salience(self) -> bool:
        return self.w_salience > 0

    @property
    def enable_novelty(self) -> bool:
        return self.w_novelty > 0


AGENTS = {
    'greedy': Agent(
        name='greedy',
        beta=0.125,   # β = 2^-3 → very precise (exploit-heavy)
        enable_extrinsic=True,
        enable_salience=False,
        enable_novelty=False,
    ),
    'random': Agent(
        name='random',
        beta=8.0,      # β = 2^3 → very imprecise (random)
        enable_extrinsic=True,
        enable_salience=False,
        enable_novelty=False,
        force_uniform=True,
    ),
    'active_learning': Agent(
        name='active_learning',
        beta=1.0,
        enable_extrinsic=True,
        enable_salience=False,
        enable_novelty=True,
    ),
    'active_inference': Agent(
        name='active_inference',
        beta=1.0,
        enable_extrinsic=True,
        enable_salience=True,
        enable_novelty=False,
    ),
    'combined': Agent(
        name='combined',
        beta=1.0,
        enable_extrinsic=True,
        enable_salience=True,
        enable_novelty=True,
    ),
}
