"""Generative model for the asymmetric T-maze (Schwartenbeck et al. 2019).

Paper-faithful setup:
  - LEFT arm is SAFE: deterministic small reward.
  - RIGHT arm is RISKY: a hidden context decides whether it pays big or nothing.
  - A cue location reveals the current context before the agent commits.

Locations: 0=centre, 1=cue, 2=left, 3=right
Hidden contexts: 0=risky_good (right pays big), 1=risky_bad (right pays none)
Observations: start | small_reward | big_reward | no_reward | cue_safe | cue_risky

Policies from centre:
    0: go directly to the left (safe) arm
    1: go directly to the right (risky) arm
    2: inspect the cue, then replan with updated beliefs
"""
import numpy as np
from numpy import ndarray


# Observation indices
OBS_START = 0
OBS_SMALL_REWARD = 1
OBS_BIG_REWARD = 2
OBS_NO_REWARD = 3
OBS_CUE_SAFE = 4       # cue says: play safe (risky arm is currently bad)
OBS_CUE_RISKY = 5      # cue says: risky arm is currently good
N_OBS = 6

# Location indices
LOC_CENTRE = 0
LOC_CUE = 1
LOC_LEFT = 2
LOC_RIGHT = 3
N_LOCATIONS = 4

# Context indices — only the *risky* arm depends on context
CTX_RISKY_GOOD = 0     # right arm pays big reward w.p. reward_prob
CTX_RISKY_BAD = 1      # right arm pays no reward w.p. reward_prob
N_CONTEXTS = 2

# Hidden state = (location, context)
N_STATES = N_LOCATIONS * N_CONTEXTS


def state_index(location: int, context: int) -> int:
    return location * N_CONTEXTS + context


def unpack_state(state: int) -> tuple[int, int]:
    return state // N_CONTEXTS, state % N_CONTEXTS


POLICY_LEFT_DIRECT = 'left_direct'
POLICY_RIGHT_DIRECT = 'right_direct'
POLICY_CUE_THEN_BEST = 'cue_then_best'
POLICY_NAMES = [POLICY_LEFT_DIRECT, POLICY_RIGHT_DIRECT, POLICY_CUE_THEN_BEST]
N_POLICIES = len(POLICY_NAMES)


class MazeModel:
    """Asymmetric T-maze generative model.

    The safe/risky structure is what makes the paper's figures interpretable:
    without a cue visit the risky arm has lower expected value than the safe
    arm (greedy always picks LEFT), but once the cue resolves the context the
    risky arm's expected value either exceeds or falls below the safe arm.
    """

    def __init__(
        self,
        reward_prob: float = 0.90,
        cue_reliability: float = 1.0,
        context_prob: float = 0.5,
        cue_cost: float = 0.6,
        stable_context: bool = False,
        no_cue: bool = False,
    ):
        self.reward_prob = reward_prob       # P(big reward | risky_good) at RIGHT arm
        self.cue_reliability = cue_reliability
        self.context_prob = context_prob     # P(context = risky_good)
        # Small opportunity cost for visiting the cue. Keeps greedy (who has
        # no salience term) from using the cue as a free detour when its
        # extrinsic value happens to tie with the safe arm under flat priors.
        self.cue_cost = cue_cost
        # Stable context: context is drawn once per experiment and the agent
        # accumulates a Dirichlet prior over contexts across trials (Fig 8B/12).
        self.stable_context = stable_context
        # No-cue variant: the cue policy is never available — the only way to
        # learn about the risky arm is to sample it directly (Fig 3/6 task).
        self.no_cue = no_cue

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
            # At centre: always see OBS_START
            self.A[OBS_START, state_index(LOC_CENTRE, ctx)] = 1.0

            # At cue: emit the cue observation matching the current context.
            s_cue = state_index(LOC_CUE, ctx)
            if ctx == CTX_RISKY_GOOD:
                self.A[OBS_CUE_RISKY, s_cue] = self.cue_reliability
                self.A[OBS_CUE_SAFE, s_cue] = 1.0 - self.cue_reliability
            else:
                self.A[OBS_CUE_SAFE, s_cue] = self.cue_reliability
                self.A[OBS_CUE_RISKY, s_cue] = 1.0 - self.cue_reliability

            # LEFT arm: deterministic small reward in every context.
            self.A[OBS_SMALL_REWARD, state_index(LOC_LEFT, ctx)] = 1.0

            # RIGHT arm: context-dependent big/none payout.
            s_right = state_index(LOC_RIGHT, ctx)
            if ctx == CTX_RISKY_GOOD:
                self.A[OBS_BIG_REWARD, s_right] = self.reward_prob
                self.A[OBS_NO_REWARD, s_right] = 1.0 - self.reward_prob
            else:
                self.A[OBS_BIG_REWARD, s_right] = 1.0 - self.reward_prob
                self.A[OBS_NO_REWARD, s_right] = self.reward_prob

    def _build_B(self):
        """Transition model: P(s' | s, target_location).

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
        """Log-preferences over observations: c = [0, 2, 4, -2, 0, 0].

        User-specified preferences: start neutral, small reward mildly liked,
        big reward strongly preferred, no reward dispreferred (same magnitude
        as the paper's loss signal), cue observations neutral.
        """
        self.c = np.array([
            0.0,             # OBS_START
            2.0,             # OBS_SMALL_REWARD
            4.0,             # OBS_BIG_REWARD
            -2.0,            # OBS_NO_REWARD
            -self.cue_cost,  # OBS_CUE_SAFE
            -self.cue_cost,  # OBS_CUE_RISKY
        ])

    def _build_d(self):
        """Prior over initial states: at centre, uncertain about context."""
        self.d = np.zeros(N_STATES)
        self.d[state_index(LOC_CENTRE, CTX_RISKY_GOOD)] = self.context_prob
        self.d[state_index(LOC_CENTRE, CTX_RISKY_BAD)] = 1.0 - self.context_prob

    def sample_context(self) -> int:
        return int(np.random.choice(
            N_CONTEXTS,
            p=[self.context_prob, 1 - self.context_prob],
        ))

    def generate_observation(self, location: int, context: int) -> int:
        s = state_index(location, context)
        return int(np.random.choice(N_OBS, p=self.A[:, s]))

    def obs_name(self, obs: int) -> str:
        return ['start', 'small', 'big', 'none', 'cue_safe', 'cue_risky'][obs]

    def loc_name(self, loc: int) -> str:
        return ['centre', 'cue', 'left', 'right'][loc]

    def ctx_name(self, ctx: int) -> str:
        return ['risky_good', 'risky_bad'][ctx]

    def policy_name(self, pi: int) -> str:
        return POLICY_NAMES[pi]

    def update_params(self, reward_prob: float | None = None,
                      context_prob: float | None = None,
                      reward_mag: float | None = None,
                      cue_cost: float | None = None) -> None:
        """Update generative-model parameters and rebuild affected matrices."""
        if reward_prob is not None:
            self.reward_prob = reward_prob
        if context_prob is not None:
            self.context_prob = context_prob
        if cue_cost is not None:
            self.cue_cost = cue_cost
        self._build_c()
        if reward_mag is not None:
            # Preserve the [+2, +4, -2] ratio at base reward_mag = 4.
            scale = reward_mag / 4.0
            self.c[OBS_SMALL_REWARD] = 2.0 * scale
            self.c[OBS_BIG_REWARD] = 4.0 * scale
            self.c[OBS_NO_REWARD] = -2.0 * scale
        self._build_A()
        self._build_d()
