"""Generative model for the retuned T-maze with cue (Schwartenbeck et al. 2019).

This is the paper's main demonstration: a 4-location maze with hidden contexts.
The agent can go directly to an arm or inspect a cue first, then replan once
that cue has resolved the current context.

Locations: 0=centre, 1=cue_arm, 2=left_arm, 3=right_arm
Hidden contexts: 0=left_rewarding, 1=right_rewarding
Observations (factored):
  - Location obs: 0=at_centre, 1=at_cue, 2=at_left, 3=at_right
  - Reward obs:   0=neutral, 1=cue_left, 2=cue_right, 3=reward, 4=loss
  (Combined into single obs index for simplicity)

Policies (initial choices from centre):
    0: go directly to the left arm
    1: go directly to the right arm
    2: inspect the cue, then replan with updated beliefs
"""
import numpy as np
from numpy import ndarray


# Observation indices (flattened)
OBS_CENTRE = 0
OBS_CUE_LEFT = 1      # cue reveals: left arm is rewarding
OBS_CUE_RIGHT = 2     # cue reveals: right arm is rewarding
OBS_REWARD = 3         # got reward
OBS_LOSS = 4           # got nothing
OBS_NEUTRAL = 5        # at arm but haven't resolved yet (step 1)
N_OBS = 6

# Location indices
LOC_CENTRE = 0
LOC_CUE = 1
LOC_LEFT = 2
LOC_RIGHT = 3
N_LOCATIONS = 4

# Context indices
CTX_LEFT_GOOD = 0     # left arm rewarding
CTX_RIGHT_GOOD = 1    # right arm rewarding
N_CONTEXTS = 2

# Hidden state = (location, context) → 4 locations × 2 contexts = 8 states
N_STATES = N_LOCATIONS * N_CONTEXTS


def state_index(location: int, context: int) -> int:
    return location * N_CONTEXTS + context


def unpack_state(state: int) -> tuple[int, int]:
    return state // N_CONTEXTS, state % N_CONTEXTS


# Policy definitions: the first decision is either a direct arm or a cue visit.
POLICY_LEFT_DIRECT = 'left_direct'
POLICY_RIGHT_DIRECT = 'right_direct'
POLICY_CUE_THEN_BEST = 'cue_then_best'
POLICY_NAMES = [POLICY_LEFT_DIRECT, POLICY_RIGHT_DIRECT, POLICY_CUE_THEN_BEST]
N_POLICIES = len(POLICY_NAMES)


class MazeModel:
    """Extended T-maze generative model with hidden contexts and cue.

    This implements the paper's primary demonstration (Figures 7-9):
    - Two hidden contexts determine which arm is rewarding
    - A cue location reveals the current context
    - Active inference agents visit the cue to reduce hidden-state uncertainty
    - Active learning agents visit uncertain arms to learn reward mappings
    """

    def __init__(
        self,
        reward_prob: float = 0.90,
        cue_reliability: float = 1.0,
        context_prob: float = 0.5,
        cue_cost: float = 0.0,
    ):
        self.reward_prob = reward_prob       # P(reward | correct arm)
        self.cue_reliability = cue_reliability  # P(correct cue | context)
        self.context_prob = context_prob     # P(context = left_good)
        self.cue_cost = cue_cost             # small opportunity cost for visiting cue

        self._build_A()
        self._build_B()
        self._build_c()
        self._build_d()

    def _build_A(self):
        """Observation model: P(observation | location, context).

        A is (N_OBS × N_STATES) where state = (location, context).
        """
        self.A = np.zeros((N_OBS, N_STATES))

        for ctx in range(N_CONTEXTS):
            # At centre: always see OBS_CENTRE
            s = state_index(LOC_CENTRE, ctx)
            self.A[OBS_CENTRE, s] = 1.0

            # At cue: see context-revealing observation
            s = state_index(LOC_CUE, ctx)
            if ctx == CTX_LEFT_GOOD:
                self.A[OBS_CUE_LEFT, s] = self.cue_reliability
                self.A[OBS_CUE_RIGHT, s] = 1.0 - self.cue_reliability
            else:
                self.A[OBS_CUE_RIGHT, s] = self.cue_reliability
                self.A[OBS_CUE_LEFT, s] = 1.0 - self.cue_reliability

            # At left arm
            s = state_index(LOC_LEFT, ctx)
            if ctx == CTX_LEFT_GOOD:
                self.A[OBS_REWARD, s] = self.reward_prob
                self.A[OBS_LOSS, s] = 1.0 - self.reward_prob
            else:
                self.A[OBS_LOSS, s] = self.reward_prob
                self.A[OBS_REWARD, s] = 1.0 - self.reward_prob

            # At right arm
            s = state_index(LOC_RIGHT, ctx)
            if ctx == CTX_RIGHT_GOOD:
                self.A[OBS_REWARD, s] = self.reward_prob
                self.A[OBS_LOSS, s] = 1.0 - self.reward_prob
            else:
                self.A[OBS_LOSS, s] = self.reward_prob
                self.A[OBS_REWARD, s] = 1.0 - self.reward_prob

    def _build_B(self):
        """Transition model: P(s' | s, target_location).

        B is (N_STATES × N_STATES × N_LOCATIONS) — one matrix per target location.
        Context does NOT change within a trial.
        """
        self.B = np.zeros((N_STATES, N_STATES, N_LOCATIONS))

        for target_loc in range(N_LOCATIONS):
            for ctx in range(N_CONTEXTS):
                for from_loc in range(N_LOCATIONS):
                    s_from = state_index(from_loc, ctx)
                    s_to = state_index(target_loc, ctx)
                    self.B[s_to, s_from, target_loc] = 1.0

    def _build_c(self):
        """Log-preferences over observations.

        Strongly prefer reward, strongly disprefer loss.
        Neutral about centre/cue observations.
        """
        self.c = np.array([
            0.0,    # OBS_CENTRE
            self.cue_cost,
            self.cue_cost,
            4.0,    # OBS_REWARD  (paper Fig 2: c_reward = +4)
            -2.0,   # OBS_LOSS    (paper Fig 2: c_loss   = -2)
            0.0,    # OBS_NEUTRAL
        ])

    def _build_d(self):
        """Prior over initial states: at centre, uncertain about context."""
        self.d = np.zeros(N_STATES)
        # At centre, split between contexts
        self.d[state_index(LOC_CENTRE, CTX_LEFT_GOOD)] = self.context_prob
        self.d[state_index(LOC_CENTRE, CTX_RIGHT_GOOD)] = 1.0 - self.context_prob

    def sample_context(self) -> int:
        """Sample a hidden context for a new trial."""
        return int(np.random.choice(N_CONTEXTS, p=[self.context_prob, 1 - self.context_prob]))

    def generate_observation(self, location: int, context: int) -> int:
        """Sample observation given true location and context."""
        s = state_index(location, context)
        probs = self.A[:, s]
        return int(np.random.choice(N_OBS, p=probs))

    def obs_name(self, obs: int) -> str:
        return ['centre', 'cue_left', 'cue_right', 'reward', 'loss', 'neutral'][obs]

    def loc_name(self, loc: int) -> str:
        return ['centre', 'cue', 'left', 'right'][loc]

    def ctx_name(self, ctx: int) -> str:
        return ['left_good', 'right_good'][ctx]

    def policy_name(self, pi: int) -> str:
        return POLICY_NAMES[pi]

    def update_params(self, reward_prob: float | None = None,
                      context_prob: float | None = None,
                      reward_mag: float | None = None,
                      cue_cost: float | None = None) -> None:
        """Update model parameters and rebuild affected matrices."""
        if reward_prob is not None:
            self.reward_prob = reward_prob
        if context_prob is not None:
            self.context_prob = context_prob
        if cue_cost is not None:
            self.cue_cost = cue_cost
        self._build_c()
        if reward_mag is not None:
            self.c[OBS_REWARD] = reward_mag
            self.c[OBS_LOSS] = -reward_mag
        self._build_A()
        self._build_d()
