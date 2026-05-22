from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BugSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    STYLE = "style"


class ImprovementType(Enum):
    PERFORMANCE = "performance"
    READABILITY = "readability"
    ROBUSTNESS = "robustness"
    CORRECTNESS = "correctness"
    STYLE = "style"


class VoteType(Enum):
    ADOPT = "adopt"
    DEFEND = "defend"
    PROPOSE_NEW = "propose_new"
    ABSTAIN = "abstain"


@dataclass
class Bug:
    description: str
    severity: BugSeverity = BugSeverity.MINOR
    line_number: int | None = None
    suggested_fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "severity": self.severity.value,
            "line_number": self.line_number,
            "suggested_fix": self.suggested_fix,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Bug":
        return cls(
            description=data["description"],
            severity=BugSeverity(data.get("severity", "minor")),
            line_number=data.get("line_number"),
            suggested_fix=data.get("suggested_fix"),
        )


@dataclass
class Improvement:
    description: str
    improvement_type: ImprovementType = ImprovementType.READABILITY
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "improvement_type": self.improvement_type.value,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Improvement":
        return cls(
            description=data["description"],
            improvement_type=ImprovementType(data.get("improvement_type", "readability")),
            priority=data.get("priority", 1),
        )


@dataclass
class Critique:
    id: str
    agent_id: str
    solution_id: str
    target_agent_id: str
    round_num: int
    overall_assessment: str = ""
    bugs: list[Bug] = field(default_factory=list)
    improvements: list[Improvement] = field(default_factory=list)
    correctness_rating: int = 5
    efficiency_rating: int = 5
    readability_rating: int = 5
    would_adopt: bool = False
    adoption_reason: str | None = None
    ratings_parsed: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def average_rating(self) -> float:
        return (self.correctness_rating + self.efficiency_rating + self.readability_rating) / 3

    @property
    def critical_bugs(self) -> list[Bug]:
        return [b for b in self.bugs if b.severity == BugSeverity.CRITICAL]

    @property
    def total_issues(self) -> int:
        return len(self.bugs) + len(self.improvements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "solution_id": self.solution_id,
            "target_agent_id": self.target_agent_id,
            "round_num": self.round_num,
            "overall_assessment": self.overall_assessment,
            "bugs": [b.to_dict() for b in self.bugs],
            "improvements": [i.to_dict() for i in self.improvements],
            "correctness_rating": self.correctness_rating,
            "efficiency_rating": self.efficiency_rating,
            "readability_rating": self.readability_rating,
            "would_adopt": self.would_adopt,
            "adoption_reason": self.adoption_reason,
            "ratings_parsed": self.ratings_parsed,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Critique":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            solution_id=data["solution_id"],
            target_agent_id=data["target_agent_id"],
            round_num=data["round_num"],
            overall_assessment=data.get("overall_assessment", ""),
            bugs=[Bug.from_dict(b) for b in data.get("bugs", [])],
            improvements=[Improvement.from_dict(i) for i in data.get("improvements", [])],
            correctness_rating=data.get("correctness_rating", 5),
            efficiency_rating=data.get("efficiency_rating", 5),
            readability_rating=data.get("readability_rating", 5),
            would_adopt=data.get("would_adopt", False),
            adoption_reason=data.get("adoption_reason"),
            ratings_parsed=data.get("ratings_parsed", True),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class Vote:
    id: str
    agent_id: str
    round_num: int
    vote_type: VoteType
    voted_solution_id: str | None = None
    voted_agent_id: str | None = None
    confidence: float = 0.5
    reasoning: str = ""
    raw_response: str = ""
    parse_failed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "round_num": self.round_num,
            "vote_type": self.vote_type.value,
            "voted_solution_id": self.voted_solution_id,
            "voted_agent_id": self.voted_agent_id,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "parse_failed": self.parse_failed,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vote":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            round_num=data["round_num"],
            vote_type=VoteType(data["vote_type"]),
            voted_solution_id=data.get("voted_solution_id"),
            voted_agent_id=data.get("voted_agent_id"),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            raw_response=data.get("raw_response", ""),
            parse_failed=data.get("parse_failed", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class ConsensusResult:
    reached: bool
    winning_solution_id: str | None = None
    winning_agent_id: str | None = None
    consensus_ratio: float = 0.0
    vote_distribution: dict[str, int] = field(default_factory=dict)
    round_num: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reached": self.reached,
            "winning_solution_id": self.winning_solution_id,
            "winning_agent_id": self.winning_agent_id,
            "consensus_ratio": self.consensus_ratio,
            "vote_distribution": self.vote_distribution,
            "round_num": self.round_num,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsensusResult":
        return cls(
            reached=data["reached"],
            winning_solution_id=data.get("winning_solution_id"),
            winning_agent_id=data.get("winning_agent_id"),
            consensus_ratio=data.get("consensus_ratio", 0.0),
            vote_distribution=data.get("vote_distribution", {}),
            round_num=data.get("round_num", 0),
            reason=data.get("reason", ""),
        )
