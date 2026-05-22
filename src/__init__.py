__version__ = "1.0.0"
__author__ = "Lukaš Patrik Magalinski"

from .models import (
    Agent,
    AgentConfig,
    AgentRole,
    Debate,
    DebateConfig,
    Solution,
    Task,
)
from .core import DebateOrchestrator
from .llm import MultiModelClient, OllamaClient

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentRole",
    "Debate",
    "DebateConfig",
    "DebateOrchestrator",
    "MultiModelClient",
    "OllamaClient",
    "Solution",
    "Task",
]
