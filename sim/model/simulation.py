"""Simulation engine: runs trials, tracks state, returns results for the UI.

This is the top-level interface called from the TypeScript bridge.
Global `sim` object is created at module load for Pyodide to call.
"""
import numpy as np
from generative_model import GenerativeModel
from dirichlet import DirichletTracker
from belief import BeliefState
from policy import evaluate_policies
from agents import AGENTS, Agent


class Simulation:
    """Manages the full simulation state for the T-maze experiment."""

    def __init__(self):
        self.model = GenerativeModel(true_reward_prob=0.75)
        self.dirichlet = DirichletTracker()
        self.belief = BeliefState(self.model.n_states)
        self.agent: Agent = AGENTS['active_learning']
        self.trial = 0
        self.history: list[dict] = []

    def reset(self, agent_type: str = 'active_learning'):
        """Reset simulation with a new agent type."""
        self.agent = AGENTS.get(agent_type, AGENTS['active_learning'])
        self.dirichlet.reset()
        self.belief.reset()
        self.trial = 0
        self.history = []

    def step(self) -> dict:
        """Run one trial: evaluate policies → choose action → observe → update beliefs.

        Returns a dict matching the StepResult interface in bridge.ts.
        """
        self.trial += 1

        # Build current A-matrix using Dirichlet expectations
        A = self.model.A_true.copy()
        A[:, 2] = self.dirichlet.get_A_column()  # replace risky column with beliefs

        # Evaluate policies
        result = evaluate_policies(
            A=A,
            c=self.model.c,
            dirichlet_conc=self.dirichlet.concentrations.copy(),
            beta=self.agent.beta,
            enable_extrinsic=self.agent.enable_extrinsic,
            enable_salience=self.agent.enable_salience,
            enable_novelty=self.agent.enable_novelty,
        )

        action = result['chosen_action']
        action_name = self.model.action_name(action)

        # Transition to new state
        new_state = action + 1  # 0→safe(1), 1→risky(2)

        # Generate observation from TRUE model
        obs = self.model.generate_observation(new_state)
        obs_name = self.model.observation_name(obs)

        # Compute reward
        reward = 0.0
        if obs == 1:    # small_reward
            reward = 1.0
        elif obs == 2:  # high_reward
            reward = 2.0
        elif obs == 3:  # no_reward
            reward = 0.0

        # Update Dirichlet if we visited the risky arm
        if new_state == 2:
            self.dirichlet.update(obs)

        # Build result dict for the UI
        step_result = {
            'trial': self.trial,
            'action': action_name,
            'observation': obs_name,
            'reward': reward,
            'beliefs': self.dirichlet.state_dict(),
            'efe': {
                'safe': result['efe_safe'],
                'risky': result['efe_risky'],
            },
            'policy_probs': result['policy_probs'],
        }

        self.history.append(step_result)
        return step_result

    def run_experiment(self, n_trials: int = 32) -> list[dict]:
        """Run multiple trials, return all results."""
        results = []
        for _ in range(n_trials):
            results.append(self.step())
        return results

    def get_config(self) -> dict:
        """Return available agent types and current parameters."""
        return {
            'agents': list(AGENTS.keys()),
            'current_agent': self.agent.name,
            'true_reward_prob': self.model.true_reward_prob,
            'trial': self.trial,
        }


# Global simulation instance for Pyodide access
sim = Simulation()
