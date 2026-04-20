"""Base class for all active inference scenarios."""
from abc import ABC, abstractmethod


class BaseScenario(ABC):
    """Interface that every scenario must implement.

    The TypeScript bridge calls these methods via Pyodide.
    All return plain dicts/lists that are JSON-serializable.
    """

    @abstractmethod
    def reset(self, agent_type: str = 'combined') -> None:
        """Reset the scenario for a new experiment."""

    @abstractmethod
    def step(self) -> dict:
        """Run one trial/step. Returns a JSON-serializable result dict."""

    @abstractmethod
    def run_experiment(self, n_steps: int = 32) -> list[dict]:
        """Run multiple steps. Returns list of step results."""

    @abstractmethod
    def get_config(self) -> dict:
        """Return scenario metadata (name, available agents, params)."""
