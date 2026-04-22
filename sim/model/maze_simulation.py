"""Simulation engine for the retuned cue-based maze.

Runs replanning trials where the agent either chooses an arm directly or visits
the cue first, then chooses an arm after the cue has updated its beliefs.
"""
import numpy as np
from numpy import ndarray

from agents import AGENTS, Agent
from math_utils import dirichlet_expected
from maze_efe import evaluate_maze_policies, select_best_arm_target
from maze_model import (
    CTX_LEFT_GOOD,
    CTX_RIGHT_GOOD,
    LOC_CENTRE,
    LOC_CUE,
    LOC_LEFT,
    LOC_RIGHT,
    MazeModel,
    N_CONTEXTS,
    N_LOCATIONS,
    N_POLICIES,
    N_STATES,
    OBS_LOSS,
    OBS_REWARD,
    POLICY_LEFT_DIRECT,
    POLICY_NAMES,
    POLICY_RIGHT_DIRECT,
    state_index,
)


class MazeSimulation:
    """Full simulation for the retuned T-maze with cue-triggered replanning."""

    def __init__(self):
        self.model = MazeModel(reward_prob=0.90, cue_reliability=1.0)
        self.agent: Agent = AGENTS['combined']
        self.trial = 0
        self.history: list[dict] = []

        # Dirichlet concentrations for reward outcomes at each arm/context state:
        # (N_LOCATIONS, N_CONTEXTS, 2) with the last axis = [reward, loss].
        self.a_conc = self._init_a_conc()
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        """Reset within-trial state."""
        self.current_context = 0
        self.current_location = LOC_CENTRE
        self.state_belief = self.model.d.copy()

    def reset(self, agent_type: str = 'combined') -> None:
        """Full reset: new agent, clear all learning."""
        self.agent = AGENTS.get(agent_type, AGENTS['combined'])
        self.trial = 0
        self.history = []
        self.a_conc = self._init_a_conc()
        self._reset_episode_state()

    def _init_a_conc(self) -> ndarray:
        """Uninformative Dirichlet prior on arm reward likelihoods.

        Following Schwartenbeck et al. (2019, Figs 5–6): every arm/context
        starts at α₀ = [1, 1], so the agent has zero prior knowledge of which
        arm pays out. Active learning then has something genuine to learn,
        and its exploration must decay as the posterior tightens.
        """
        return np.ones((N_LOCATIONS, N_CONTEXTS, 2))

    def step(self) -> dict:
        """Run one replanning trial and return the full trajectory."""
        self.trial += 1
        self._reset_episode_state()
        self.current_context = self.model.sample_context()

        A_believed = self._believed_A()
        result = evaluate_maze_policies(
            A=A_believed,
            B=self.model.B,
            c=self.model.c,
            d=self.state_belief,
            a_conc=self.a_conc,
            beta=self.agent.beta,
            w_extrinsic=self.agent.w_extrinsic,
            w_salience=self.agent.w_salience,
            w_novelty=self.agent.w_novelty,
            force_uniform=self.agent.force_uniform,
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
                a_conc=self.a_conc,
                w_extrinsic=self.agent.w_extrinsic,
                w_salience=self.agent.w_salience,
                w_novelty=self.agent.w_novelty,
                force_uniform=self.agent.force_uniform,
            )
            obs = self._execute_visit(target_loc, A_believed, trajectory)
            reward = self._reward_after_obs(reward, obs)

        efe_summary = {}
        for pi in range(N_POLICIES):
            efe_summary[POLICY_NAMES[pi]] = result['efe'][pi]

        step_result = {
            'trial': self.trial,
            'context': self.model.ctx_name(self.current_context),
            'policy': policy_name,
            'trajectory': trajectory,
            'reward': reward,
            'beliefs': {
                'context_belief': self._context_belief().tolist(),
                'left_arm': self._arm_summary(LOC_LEFT),
                'right_arm': self._arm_summary(LOC_RIGHT),
            },
            'efe': efe_summary,
            'policy_probs': {POLICY_NAMES[i]: result['probs'][i] for i in range(N_POLICIES)},
        }

        self.history.append(step_result)
        return step_result

    def _execute_visit(self, target_loc: int, A_believed: ndarray, trajectory: list[dict]) -> int:
        """Move to a location, then condition on the resulting observation."""
        self.current_location = target_loc
        self._transition_belief(target_loc)

        obs = self.model.generate_observation(target_loc, self.current_context)
        trajectory.append({
            'step': len(trajectory),
            'location': self.model.loc_name(target_loc),
            'observation': self.model.obs_name(obs),
        })
        self._bayesian_update(obs, A_believed)

        # Only agents that *value* parameter learning (novelty-driven) commit
        # observations to α. The paper's greedy and active-inference agents
        # plan against a fixed model — tracking a posterior they don't consult
        # would just muddy the leaderboard. This keeps the Fig 6 contrast
        # crisp: without the novelty drive, there is no model learning.
        if target_loc in (LOC_LEFT, LOC_RIGHT) and self.agent.w_novelty > 0:
            outcome_idx = None
            if obs == OBS_REWARD:
                outcome_idx = 0
            elif obs == OBS_LOSS:
                outcome_idx = 1
            if outcome_idx is not None:
                self.a_conc[target_loc, :, outcome_idx] += self._context_belief()

        return obs

    def _transition_belief(self, target_loc: int) -> None:
        """Apply the location transition before conditioning on the observation."""
        next_belief = self.model.B[:, :, target_loc] @ self.state_belief
        total = next_belief.sum()
        if total > 1e-16:
            self.state_belief = next_belief / total

    def _reward_after_obs(self, reward: float, obs: int) -> float:
        """Convert the latest observation into the per-trial reward summary.

        Magnitudes mirror the agent's preferences c = [+4, -2] from
        Schwartenbeck et al. (2019, Fig 2), scaled to a unit reward.
        """
        if obs == OBS_REWARD:
            return 1.0
        if obs == OBS_LOSS:
            return -0.5
        return reward

    def _bayesian_update(self, obs: int, A: ndarray) -> None:
        """Update state belief given observation."""
        likelihood = A[obs, :]
        unnorm = likelihood * self.state_belief
        total = unnorm.sum()
        if total > 1e-16:
            self.state_belief = unnorm / total

    def _believed_A(self) -> ndarray:
        """Return the observation model the agent uses for planning.

        Following Schwartenbeck et al. (2019, Figs 5–6): the cue and location
        likelihoods are known, but the agent has a flat Dirichlet prior over
        arm reward reliabilities and must learn them from experience. We plug
        the Dirichlet expectation into the REWARD/LOSS rows of A so that
        extrinsic and salience are computed against the *learned* model —
        novelty lives in the same α concentrations. Agents that never update α
        (greedy, active_inference) plan forever as if reward_prob = 0.5,
        which is what lets active learning pull ahead on this task.
        """
        A = self.model.A.copy()
        for loc in (LOC_LEFT, LOC_RIGHT):
            for ctx in range(N_CONTEXTS):
                s = state_index(loc, ctx)
                conc = self.a_conc[loc, ctx]
                total = float(conc[0] + conc[1])
                if total <= 0:
                    continue
                p_reward = float(conc[0]) / total
                A[OBS_REWARD, s] = p_reward
                A[OBS_LOSS, s] = 1.0 - p_reward
        return A

    def _arm_summary(self, arm_loc: int) -> dict:
        """Expose a context-weighted arm summary for the UI belief bars."""
        ctx_belief = self._context_belief()
        conc_reward = 0.0
        conc_loss = 0.0
        p_reward = 0.0

        for context in range(N_CONTEXTS):
            weight = float(ctx_belief[context])
            conc_reward += weight * float(self.a_conc[arm_loc, context, 0])
            conc_loss += weight * float(self.a_conc[arm_loc, context, 1])
            p_reward += weight * float(dirichlet_expected(self.a_conc[arm_loc, context])[0])

        return {
            'conc_reward': conc_reward,
            'conc_loss': conc_loss,
            'p_reward': p_reward,
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
        results = []
        for _ in range(n_trials):
            results.append(self.step())
        return results

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
