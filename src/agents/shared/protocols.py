"""Agent protocols and interfaces"""

from typing import Protocol, runtime_checkable
from src.agents.shared.state import AgentState


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol that all agents must implement"""

    async def can_handle(self, state: AgentState) -> bool:
        """Check if agent can handle this request"""
        ...

    async def execute(self, state: AgentState) -> AgentState:
        """Execute agent logic and update state"""
        ...

    def get_name(self) -> str:
        """Get agent name"""
        ...
