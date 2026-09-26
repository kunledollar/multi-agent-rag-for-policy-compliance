"""Base interface for autonomous Sentinel agents."""
from abc import ABC, abstractmethod

from .contracts import AgentDecision, SentinelState


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, state: SentinelState) -> AgentDecision:
        """Inspect shared state and return an explicit, explainable decision."""
        raise NotImplementedError

