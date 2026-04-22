"""Simulation engine for the asymmetric T-maze.

Runs replanning trials where the agent either commits directly to an arm or
inspects the cue first and then picks an arm with the updated belief.
"""
import numpy as np
from numpy import ndarray

from agents import AGENTS, Agent
from math_utils import dirichlet_expected
from maze_efe import evaluate_maze_policies, select_best_arm_target
from maze_model import (
    CTX_RISKY_BAD,
    CTX_RISKY_GOOD,
    LOC_CENTRE,
    LOC_CUE,
    LOC_LEFT,
    LOC_RIGHT,
    MazeModel,
    N_CONTEXTS,
    N_POLICIES,
    N_STATES,
    OBS_BIG_REWARD,
    OBS_CUE_RISKY,
    OBS_CUE_SAFE,
    OBS_NO_REWARD,
    OBS_SMALL_REWARD,
    POLICY_LEFT_DIRECT,
    POLICY_NAMES,
    POLICY_RIGHT_DIRECT,
    state_index,
)


class MazeSimulation:
    """Asymmetric T-maze with cue-triggered replanning."""

    def __init__(self, stable_context: bool = False, no_cue: bool = False):
        self.model = MazeModel(
            reward_prob=0.90,
            cue_reliability=1.0,
            stable_context=stable_context,
            no_cue=no_cue,
        )
        self.agent: Agent = AGENTS['combined']
        self.trial = 0
        self.history: list[dict] = []

        # Only the RIGHT arm has learnable reliability — LEFT is known to be
        # deterministic. Shape: (N_CONTEXTS, 2) for [big_reward, no_reward].
        self.right_conc = self._init_right_conc()
        # Dirichlet prior over contexts (d-vector). In volatile mode this is
        # never updated — the context really does resample every trial, so the
        # agent's prior stays uniform. In stable mode, cue observations update
        # it across trials, collapsing the cue's info gain (Fig 8B / Fig 8C).
        self.d_conc = self._init_d_conc()
        self.fixed_context = 0
        self._reset_episode_state(new_episode=True)

    def _reset_episode_state(self, new_episode: bool = False) -> None:
        """Reset per-trial location/belief. In stable mode the agent's belief
        about context is *carried over* between trials via d_conc, so only the
        location component of the belief is refreshed."""
        self.current_location = LOC_CENTRE
        if self.model.stable_context:
            d_expected = dirichlet_expected(self.d_conc)
            belief = np.zeros(N_STATES)
            for ctx in range(N_CONTEXTS):
                belief[state_index(LOC_CENTRE, ctx)] = d_expected[ctx]
            self.state_belief = belief
            if new_episode:
                # New experiment — draw one context that holds for the whole run.
                self.fixed_context = self.model.sample_context()
            self.current_context = self.fixed_context
        else:
            self.current_context = 0
            self.state_belief = self.model.d.copy()

    def reset(self, agent_type: str = 'combined') -> None:
        """Full reset: new agent, clear all learned α and d."""
        self.agent = AGENTS.get(agent_type, AGENTS['combined'])
        self.trial = 0
        self.history = []
        self.right_conc = self._init_right_conc()
        self.d_conc = self._init_d_conc()
        self._reset_episode_state(new_episode=True)

    def _init_d_conc(self) -> ndarray:
        """Uniform Dirichlet prior over the two contexts.

        Mirrors the paper's setup where the agent starts each experiment
        with no prior knowledge about which context is active. In stable
        mode, this concentrates as the cue repeatedly signals the same
        context — driving the hidden-state uncertainty to zero and
        collapsing the cue's salience (Fig 8B).
        """
        return np.ones(N_CONTEXTS)

    def _init_right_conc(self) -> ndarray:
        """Dirichlet prior on the risky arm's reward distribution.

        Two regimes:
          * Cue-based tasks (``no_cue=False``): the agent knows the task
            *structure* — which context favours which outcome. Encoded as
            α[risky_good] = [4, 1] (initial p_big ≈ 0.80) and its mirror.
            Weaker priors (e.g. [1, 0.25]) collapse under a single unlucky
            draw and the agent never recovers.
          * Active-learning task (``no_cue=True``, Fig 3/6): the agent
            starts fully ignorant — α=[1, 1] so the novelty term is large
            and drives the iconic early-sampling spike of Fig 6C.
        """
        conc = np.zeros((N_CONTEXTS, 2))
        if self.model.no_cue:
            # Flat prior in both contexts. Since the cue is disabled and the
            # context is stable, the agent effectively learns a single
            # Dirichlet over (big, none) — mimicking the paper's Task-A
            # parameter-exploration setting.
            conc[CTX_RISKY_GOOD] = [1.0, 1.0]
            conc[CTX_RISKY_BAD] = [1.0, 1.0]
        else:
            conc[CTX_RISKY_GOOD] = [4.0, 1.0]
            conc[CTX_RISKY_BAD] = [1.0, 4.0]
        return conc

    def _a_conc_for_efe(self) -> ndarray:
        """Pack α into the (N_LOCATIONS, N_CONTEXTS, 2) tensor EFE expects.

        LEFT arm entries are never consulted because novelty is only computed
        for RIGHT, but we still populate them with deterministic [≫1, ε]
        values so the full tensor is well-formed.
        """
        a_conc = np.ones((4, N_CONTEXTS, 2))
        # LEFT arm: near-deterministic small reward (not learned).
        a_conc[LOC_LEFT, :, 0] = 1e6
        a_conc[LOC_LEFT, :, 1] = 1e-6
        a_conc[LOC_RIGHT] = self.right_conc
        return a_conc

    def step(self) -> dict:
        """Run one replanning trial and return the full trajectory."""
        self.trial += 1
        self._reset_episode_state(new_episode=False)
        if not self.model.stable_context:
            self.current_context = self.model.sample_context()

        A_believed = self._believed_A()
        a_conc = self._a_conc_for_efe()
        result = evaluate_maze_policies(
            A=A_believed,
            B=self.model.B,
            c=self.model.c,
            d=self.state_belief,
            a_conc=a_conc,
            beta=self.agent.beta,
            w_extrinsic=self.agent.w_extrinsic,
            w_salience=self.agent.w_salience,
            w_novelty=self.agent.w_novelty,
            force_uniform=self.agent.force_uniform,
            no_cue=self.model.no_cue,
        )

        policy_name = POLICY_NAMES[result['chosen_policy']]
        trajectory: list[dict] = []
        reward = 0.0

        if policy_name == POLICY_LEFT_DIRECT:
            obs = self._execute_visit(LOC_LEFT, A_believed, trajectory)
            reward = self._reward_after_obs(reward, obs)
        elif policy_name == POLICY_RIGHT_DIRECT:
            obs = self._execute_visit(LOC_RIGHT, A_believed, trajectory)
            reward = self._reward_after_obs(reward, obs)
        else:
            obs = self._execute_visit(LOC_CUE, A_believed, trajectory)
            reward = self._reward_after_obs(reward, obs)

            target_loc, _ = select_best_arm_target(
                A=A_believed,
                B=self.model.B,
                c=self.model.c,
                belief=self.state_belief,
                a_conc=a_conc,
                w_extrinsic=self.agent.w_extrinsic,
                w_salience=self.agent.w_salience,
                w_novelty=self.agent.w_novelty,
                force_uniform=self.agent.force_uniform,
            )
            obs = self._execute_visit(target_loc, A_believed, trajectory)
            reward = self._reward_after_obs(reward, obs)

        efe_summary = {
            POLICY_NAMES[pi]: result['efe'][pi] for pi in range(N_POLICIES)
        }

        step_result = {
            'trial': self.trial,
            'context': self.model.ctx_name(self.current_context),
            'policy': policy_name,
            'trajectory': trajectory,
            'reward': reward,
            'beliefs': {
                'context_belief': self._context_belief().tolist(),
                'left_arm': self._left_arm_summary(),
                'right_arm': self._right_arm_summary(),
            },
            'efe': efe_summary,
            'policy_probs': {POLICY_NAMES[i]: result['probs'][i] for i in range(N_POLICIES)},
        }

        self.history.append(step_result)
        return step_result

    def _execute_visit(self, target_loc: int, A_believed: ndarray, trajectory: list[dict]) -> int:
        """Move to a location, observe, and update beliefs."""
        self.current_location = target_loc
        self._transition_belief(target_loc)

        obs = self.model.generate_observation(target_loc, self.current_context)
        trajectory.append({
            'step': len(trajectory),
            'location': self.model.loc_name(target_loc),
            'observation': self.model.obs_name(obs),
        })
        self._bayesian_update(obs, A_believed)

        # Only agents that value parameter learning (novelty-driven) update α.
        # LEFT arm is deterministic, so even they only learn at RIGHT.
        if target_loc == LOC_RIGHT and self.agent.w_novelty > 0:
            outcome_idx = None
            if obs == OBS_BIG_REWARD:
                outcome_idx = 0
            elif obs == OBS_NO_REWARD:
                outcome_idx = 1
            if outcome_idx is not None:
                self.right_conc[:, outcome_idx] += self._context_belief()

        # Stable-context d-vector learning: cue observations update the
        # agent's prior over contexts across trials. This is the mechanism
        # behind Fig 8B — after a few consistent cues, the prior is so
        # concentrated that the cue's expected information gain collapses
        # and the active-inference agent stops visiting it.
        if self.model.stable_context and target_loc == LOC_CUE:
            if obs == OBS_CUE_RISKY:
                self.d_conc[CTX_RISKY_GOOD] += 1.0
            elif obs == OBS_CUE_SAFE:
                self.d_conc[CTX_RISKY_BAD] += 1.0

        return obs

    def _transition_belief(self, target_loc: int) -> None:
        next_belief = self.model.B[:, :, target_loc] @ self.state_belief
        total = next_belief.sum()
        if total > 1e-16:
            self.state_belief = next_belief / total

    def _reward_after_obs(self, reward: float, obs: int) -> float:
        """Map observations to a numeric reward signal for plotting.

        Aligned with user's c = [0, 2, 4, -2, 0, 0] but scaled to [-0.5, 1.0]
        for the UI chart. ``no_reward`` yields 0.0 — the agent still disprefers
        it via c, but there is no loss payout in the outcome itself.
        """
        if obs == OBS_BIG_REWARD:
            return 1.0
        if obs == OBS_SMALL_REWARD:
            return 0.5
        if obs == OBS_NO_REWARD:
            return 0.0
        return reward

    def _bayesian_update(self, obs: int, A: ndarray) -> None:
        likelihood = A[obs, :]
        unnorm = likelihood * self.state_belief
        total = unnorm.sum()
        if total > 1e-16:
            self.state_belief = unnorm / total

    def _believed_A(self) -> ndarray:
        """Agent's observation model with learned RIGHT-arm reward probs.

        LEFT arm rows stay fixed (the agent knows it's safe). RIGHT arm rows
        are overridden by the Dirichlet expectation in each context so that
        extrinsic and salience reflect the *learned* model. Novelty lives in
        the same α. Agents that never update α plan with flat priors and so
        can never pull ahead of the safe arm.
        """
        A = self.model.A.copy()
        for ctx in range(N_CONTEXTS):
            s = state_index(LOC_RIGHT, ctx)
            conc = self.right_conc[ctx]
            total = float(conc[0] + conc[1])
            if total <= 0:
                continue
            p_big = float(conc[0]) / total
            A[OBS_BIG_REWARD, s] = p_big
            A[OBS_NO_REWARD, s] = 1.0 - p_big
        return A

    def _left_arm_summary(self) -> dict:
        """LEFT arm is deterministic — expose it as a constant."""
        return {
            'p_reward': 1.0,
            'conc_reward': 0.0,
            'conc_loss': 0.0,
            'deterministic': True,
        }

    def _right_arm_summary(self) -> dict:
        """Context-weighted summary of the risky arm's learned belief."""
        ctx_belief = self._context_belief()
        conc_big = 0.0
        conc_none = 0.0
        p_big = 0.0
        for ctx in range(N_CONTEXTS):
            w = float(ctx_belief[ctx])
            conc_big += w * float(self.right_conc[ctx, 0])
            conc_none += w * float(self.right_conc[ctx, 1])
            p_big += w * float(dirichlet_expected(self.right_conc[ctx])[0])
        return {
            'p_reward': p_big,
            'conc_reward': conc_big,
            'conc_loss': conc_none,
            'deterministic': False,
        }

    def _context_belief(self) -> ndarray:
        """Marginalize state belief to get P(context)."""
        ctx_belief = np.zeros(N_CONTEXTS)
        for state in range(N_STATES):
            _, context = divmod(state, N_CONTEXTS)
            ctx_belief[context] += self.state_belief[state]
        total = ctx_belief.sum()
        if total > 1e-16:
            ctx_belief /= total
        return ctx_belief

    def run_experiment(self, n_trials: int = 32) -> list[dict]:
        return [self.step() for _ in range(n_trials)]

    def get_config(self) -> dict:
        return {
            'agents': list(AGENTS.keys()),
            'current_agent': self.agent.name,
            'reward_prob': self.model.reward_prob,
            'cue_reliability': self.model.cue_reliability,
            'trial': self.trial,
        }


# Global instance for Pyodide
sim = MazeSimulation()
