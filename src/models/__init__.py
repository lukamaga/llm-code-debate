"""
Data models for the LLM Code Debate System.
"""
from .agent import (
    Agent,
    AgentConfig,
    AgentMessage,
    AgentRole,
    AgentStats,
)
from .solution import (
    CodeQualityMetrics,
    ExecutionResult,
    Solution,
    SolutionStatus,
    Task,
    TestResult,
)
from .critique import (
    Bug,
    BugSeverity,
    ConsensusResult,
    Critique,
    Improvement,
    ImprovementType,
    Vote,
    VoteType,
)
from .debate import (
    Debate,
    DebateConfig,
    DebateStatus,
    RoundSummary,
)
from .metrics import (
    AgentProfile,
    DebateMetrics,
    ExperimentSummary,
)

__all__ = [
    # Agent
    "Agent",
    "AgentConfig",
    "AgentMessage",
    "AgentRole",
    "AgentStats",
    # Solution
    "CodeQualityMetrics",
    "ExecutionResult",
    "Solution",
    "SolutionStatus",
    "Task",
    "TestResult",
    # Critique
    "Bug",
    "BugSeverity",
    "ConsensusResult",
    "Critique",
    "Improvement",
    "ImprovementType",
    "Vote",
    "VoteType",
    # Debate
    "Debate",
    "DebateConfig",
    "DebateStatus",
    "RoundSummary",
    # Metrics
    "AgentProfile",
    "DebateMetrics",
    "ExperimentSummary",
]
