"""Base agent class for all domain agents"""

from abc import ABC, abstractmethod
from typing import Optional
import time
from src.agents.shared.state import AgentState
from src.utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all domain agents"""

    def __init__(self):
        """Initialize base agent"""
        self.logger = get_logger(self.__class__.__name__)
        self._start_time = None

    @abstractmethod
    async def can_handle(self, state: AgentState) -> bool:
        """
        Check if this agent can handle the current request

        Args:
            state: Current agent state

        Returns:
            True if agent can handle, False otherwise
        """
        pass

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute agent logic and update state

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get agent name

        Returns:
            Agent name string
        """
        pass

    def _update_metadata(self, state: AgentState, key: str, value: any) -> None:
        """Helper to update state metadata"""
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"][key] = value

    def _log_start(self, operation: str) -> None:
        """Log operation start"""
        self._start_time = time.time()
        self.logger.info(f"{self.get_name()}: {operation} started")

    def _log_complete(self, operation: str, **metrics) -> None:
        """Log operation complete with timing"""
        duration_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        if metrics_str:
            self.logger.info(f"{self.get_name()}: {operation} completed in {duration_ms:.0f}ms ({metrics_str})")
        else:
            self.logger.info(f"{self.get_name()}: {operation} completed in {duration_ms:.0f}ms")

    def _log_error(self, operation: str, error: Exception) -> None:
        """Log operation error with timing"""
        duration_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0
        self.logger.error(f"{self.get_name()}: {operation} failed after {duration_ms:.0f}ms - {str(error)}")
