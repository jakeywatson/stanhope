"""Dirichlet concentration parameter tracking for the observation model.

The agent learns the A-matrix (specifically the risky arm column) by
maintaining Dirichlet concentration parameters that are updated with
each observation.
"""
import numpy as np
from numpy import ndarray
from math_utils import dirichlet_expected


class DirichletTracker:
    """Tracks Dirichlet concentration parameters for the risky arm.

    The risky arm column of the A-matrix has two relevant entries:
      - a_high: concentration for P(high_reward | risky)
      - a_none: concentration for P(no_reward | risky)

    Prior: a_high = a_none = 1.0 (uniform — maximum uncertainty)
    """

    def __init__(self, prior_high: float = 1.0, prior_none: float = 1.0):
        self.prior = np.array([prior_high, prior_none])
        self.concentrations = self.prior.copy()

    def update(self, observation: int):
        """Update concentrations after observing outcome in risky arm.

        observation: 2 = high_reward → increment a_high
                     3 = no_reward  → increment a_none
        """
        if observation == 2:    # high_reward
            self.concentrations[0] += 1.0
        elif observation == 3:  # no_reward
            self.concentrations[1] += 1.0

    def expected_probs(self) -> ndarray:
        """Expected reward probabilities: E[Dir(α)]."""
        return dirichlet_expected(self.concentrations)

    def p_high_reward(self) -> float:
        """Expected probability of high reward in risky arm."""
        return self.expected_probs()[0]

    def total_counts(self) -> float:
        """Total concentration (certainty indicator)."""
        return self.concentrations.sum()

    def get_A_column(self) -> ndarray:
        """Return the full 4-element risky arm column for the A-matrix.

        [P(at_start|risky), P(small_reward|risky), P(high_reward|risky), P(no_reward|risky)]
        = [0, 0, E[p_high], E[p_none]]
        """
        p = self.expected_probs()
        return np.array([0.0, 0.0, p[0], p[1]])

    def reset(self):
        self.concentrations = self.prior.copy()

    def state_dict(self) -> dict:
        return {
            'conc_high': float(self.concentrations[0]),
            'conc_none': float(self.concentrations[1]),
            'p_high': float(self.p_high_reward()),
        }
