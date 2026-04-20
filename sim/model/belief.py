"""Posterior belief tracking over hidden states."""
import numpy as np
from numpy import ndarray
from math_utils import softmax, log_stable


class BeliefState:
    """Tracks the agent's posterior belief over hidden states.

    In the simple T-maze, the agent always starts at state 0 (start position),
    so the belief is trivial. This class exists for the extended maze with
    hidden contexts (Phase 7 extension).
    """

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.posterior = np.zeros(n_states)
        self.posterior[0] = 1.0  # certain: at start

    def update(self, observation: int, A: ndarray):
        """Bayesian belief update: P(s|o) ∝ P(o|s) P(s)."""
        likelihood = A[observation, :]
        unnorm = likelihood * self.posterior
        total = unnorm.sum()
        if total > 0:
            self.posterior = unnorm / total
        else:
            self.posterior = np.ones(self.n_states) / self.n_states

    def set_state(self, state: int):
        """Set belief to certainty about a specific state."""
        self.posterior = np.zeros(self.n_states)
        self.posterior[state] = 1.0

    def reset(self):
        self.posterior = np.zeros(self.n_states)
        self.posterior[0] = 1.0
