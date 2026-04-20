"""Generative model for the T-maze task (Schwartenbeck et al. 2019).

Defines the A-matrix (observations|states), B-matrix (transitions|actions),
c-vector (preferences), and d-vector (initial state prior).

States:  0=start, 1=safe_arm, 2=risky_arm
Observations: 0=at_start, 1=small_reward, 2=high_reward, 3=no_reward
Actions: 0=stay/go_safe, 1=go_risky
"""
import numpy as np
from numpy import ndarray


class GenerativeModel:
    """Parameterised generative model for the T-maze.

    The A-matrix for the risky arm column is *learned* via Dirichlet updates.
    Everything else is fixed.
    """

    def __init__(self, true_reward_prob: float = 0.75):
        self.n_states = 3       # start, safe, risky
        self.n_obs = 4          # at_start, small_reward, high_reward, no_reward
        self.n_actions = 2      # go_safe, go_risky

        # True reward probability for the risky arm (used to generate observations)
        self.true_reward_prob = true_reward_prob

        # --- Observation model A (4 obs × 3 states) ---
        # Columns: start, safe, risky
        # Rows: at_start, small_reward, high_reward, no_reward
        #
        # Start and safe columns are deterministic.
        # Risky column depends on true_reward_prob (for generating data)
        # but the agent's *belief* about it comes from Dirichlet concentrations.
        self.A_true = np.array([
            [1.0, 0.0, 0.0],                           # at_start
            [0.0, 1.0, 0.0],                           # small_reward
            [0.0, 0.0, true_reward_prob],               # high_reward
            [0.0, 0.0, 1.0 - true_reward_prob],        # no_reward
        ])

        # --- Transition model B (3 states × 3 states × 2 actions) ---
        # B[:, :, a] = transition matrix given action a
        # From start: action 0 → safe, action 1 → risky
        # From safe/risky: stay (absorbing for the trial)
        self.B = np.zeros((self.n_states, self.n_states, self.n_actions))
        # Action 0: go safe
        self.B[1, 0, 0] = 1.0  # start → safe
        self.B[1, 1, 0] = 1.0  # safe → safe (absorbing)
        self.B[2, 2, 0] = 1.0  # risky → risky (absorbing)
        # Action 1: go risky
        self.B[2, 0, 1] = 1.0  # start → risky
        self.B[1, 1, 1] = 1.0  # safe → safe (absorbing)
        self.B[2, 2, 1] = 1.0  # risky → risky (absorbing)

        # --- Preferences c (log-preferences over observations) ---
        # c = [0, 2, 4, -2] from paper
        self.c = np.array([0.0, 2.0, 4.0, -2.0])

        # --- Initial state prior d ---
        self.d = np.array([1.0, 0.0, 0.0])  # always start at position 0

    def generate_observation(self, state: int) -> int:
        """Sample an observation given the true state."""
        probs = self.A_true[:, state]
        return int(np.random.choice(self.n_obs, p=probs))

    def observation_name(self, obs: int) -> str:
        return ['at_start', 'small_reward', 'high_reward', 'no_reward'][obs]

    def state_name(self, state: int) -> str:
        return ['start', 'safe', 'risky'][state]

    def action_name(self, action: int) -> str:
        return ['safe', 'risky'][action]
