"""
Debate models for the LLM Code Debate System.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .agent import Agent, AgentStats
from .solution import Solution, Task, ExecutionResult
from .critique import Critique, Vote, ConsensusResult


class DebateStatus(Enum):
    """Status of the debate."""
    PENDING = "pending"
    RUNNING = "running"
    CONSENSUS_REACHED = "consensus_reached"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    EARLY_STOP = "early_stop"  # All tests passed early
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class RoundSummary:
    """Summary of a single debate round."""
    round_num: int
    solutions: list[Solution]
    critiques: list[Critique]
    votes: list[Vote]
    consensus_result: ConsensusResult | None = None
    
    # Stats
    best_pass_rate: float = 0.0
    avg_pass_rate: float = 0.0
    bugs_found: int = 0
    improvements_suggested: int = 0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    
    def compute_stats(self) -> None:
        """Compute summary statistics for the round."""
        if self.solutions:
            pass_rates = [s.pass_rate for s in self.solutions]
            self.best_pass_rate = max(pass_rates) if pass_rates else 0.0
            self.avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        
        self.bugs_found = sum(len(c.bugs) for c in self.critiques)
        self.improvements_suggested = sum(len(c.improvements) for c in self.critiques)
        
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "round_num": self.round_num,
            "solutions": [s.to_dict() for s in self.solutions],
            "critiques": [c.to_dict() for c in self.critiques],
            "votes": [v.to_dict() for v in self.votes],
            "consensus_result": self.consensus_result.to_dict() if self.consensus_result else None,
            "best_pass_rate": self.best_pass_rate,
            "avg_pass_rate": self.avg_pass_rate,
            "bugs_found": self.bugs_found,
            "improvements_suggested": self.improvements_suggested,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Debate:
    """
    A complete debate session.
    
    Contains all agents, rounds, and final results.
    """
    id: str
    task: Task
    agents: list[Agent]
    rounds: list[RoundSummary] = field(default_factory=list)
    
    # Status
    status: DebateStatus = DebateStatus.PENDING
    current_round: int = 0
    
    # Configuration
    max_rounds: int = 5
    consensus_threshold: float = 0.6
    
    # Results
    final_solution: Solution | None = None
    final_consensus: ConsensusResult | None = None
    winning_agent_id: str | None = None
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    
    # Error tracking
    error_message: str | None = None
    
    @property
    def duration_seconds(self) -> float:
        """Total debate duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def total_rounds(self) -> int:
        """Number of completed rounds."""
        return len(self.rounds)
    
    @property
    def all_solutions(self) -> list[Solution]:
        """Get all solutions from all rounds."""
        solutions = []
        for round_summary in self.rounds:
            solutions.extend(round_summary.solutions)
        return solutions
    
    @property
    def all_critiques(self) -> list[Critique]:
        """Get all critiques from all rounds."""
        critiques = []
        for round_summary in self.rounds:
            critiques.extend(round_summary.critiques)
        return critiques
    
    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_latest_solutions(self) -> list[Solution]:
        """Get the most recent solution from each agent."""
        if not self.rounds:
            return []
        return self.rounds[-1].solutions
    
    def get_agent_stats(self) -> list[AgentStats]:
        """Get statistics for all agents."""
        return [agent.stats for agent in self.agents]
    
    def add_round(self, round_summary: RoundSummary) -> None:
        """Add a completed round."""
        round_summary.compute_stats()
        self.rounds.append(round_summary)
        self.current_round = round_summary.round_num
    
    def finalize(
        self,
        status: DebateStatus,
        final_solution: Solution | None = None,
        consensus: ConsensusResult | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finalize the debate with results."""
        self.status = status
        self.final_solution = final_solution
        self.final_consensus = consensus
        self.error_message = error_message
        self.end_time = datetime.now()
        
        if consensus and consensus.winning_agent_id:
            self.winning_agent_id = consensus.winning_agent_id
            # Update winner's stats
            winner = self.get_agent(consensus.winning_agent_id)
            if winner:
                winner.stats.times_won_debate += 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task.to_dict(),
            "agents": [a.to_dict() for a in self.agents],
            "rounds": [r.to_dict() for r in self.rounds],
            "status": self.status.value,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            "final_solution": self.final_solution.to_dict() if self.final_solution else None,
            "final_consensus": self.final_consensus.to_dict() if self.final_consensus else None,
            "winning_agent_id": self.winning_agent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class DebateConfig:
    """Configuration for a debate."""
    max_rounds: int = 5
    min_rounds: int = 2
    consensus_threshold: float = 0.6
    early_stop_on_perfect: bool = True
    temperature_initial: float = 0.3
    temperature_critique: float = 0.5
    temperature_revision: float = 0.4
    execution_timeout: int = 30
    revision_show_all_solutions: bool = False
    # "uniform" = all agents get same prompt (baseline), "diverse" = DMAD-style strategies
    revision_strategy: str = "uniform"
    # Per-agent strategy overrides: {"agent_1_mistral": "simplify", ...}
    # Empty dict = auto-assign round-robin when revision_strategy="diverse"
    agent_strategies: dict[str, str] = field(default_factory=dict)
    # Increase revision temperature when pass_rate stagnates between rounds
    adaptive_temperature: bool = False
    # Include critique history from previous rounds in prompts
    critique_history: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebateConfig:
        return cls(
            max_rounds=data.get("max_rounds", 5),
            min_rounds=data.get("min_rounds", 2),
            consensus_threshold=data.get("consensus_threshold", 0.6),
            early_stop_on_perfect=data.get("early_stop_on_perfect", True),
            temperature_initial=data.get("temperature_initial", 0.3),
            temperature_critique=data.get("temperature_critique", 0.5),
            temperature_revision=data.get("temperature_revision", 0.4),
            execution_timeout=data.get("execution_timeout", 30),
            revision_show_all_solutions=data.get("revision_show_all_solutions", False),
            revision_strategy=data.get("revision_strategy", "uniform"),
            agent_strategies=data.get("agent_strategies", {}),
            adaptive_temperature=data.get("adaptive_temperature", False),
            critique_history=data.get("critique_history", False),
        )
