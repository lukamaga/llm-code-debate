from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime


class AgentRole(Enum):
    GENERAL = "general"
    PROPOSER = "proposer"
    CRITIC = "critic"
    OPTIMIZER = "optimizer"
    JUDGE = "judge"


@dataclass
class AgentConfig:
    name: str
    model: str
    role: AgentRole = AgentRole.GENERAL
    temperature: float = 0.3
    max_tokens: int = 12288

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        return cls(
            name=data.get("name", data.get("model", "unknown")),
            model=data.get("model", data.get("name", "")),
            role=AgentRole(data.get("role", "general")),
            temperature=data.get("temperature", 0.3),
            max_tokens=data.get("max_tokens", 12288),
        )


@dataclass
class AgentMessage:
    agent_id: str
    round_num: int
    message_type: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStats:
    agent_id: str
    model: str
    role: AgentRole
    
    solutions_proposed: int = 0
    solutions_revised: int = 0
    
    critiques_given: int = 0
    critiques_received: int = 0
    bugs_found: int = 0
    improvements_suggested: int = 0
    
    times_changed_mind: int = 0
    times_defended: int = 0
    times_adopted_other: int = 0
    
    times_won_debate: int = 0
    final_votes_received: int = 0
    
    total_generation_time: float = 0.0
    avg_response_time: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "role": self.role.value,
            "solutions_proposed": self.solutions_proposed,
            "solutions_revised": self.solutions_revised,
            "critiques_given": self.critiques_given,
            "critiques_received": self.critiques_received,
            "bugs_found": self.bugs_found,
            "improvements_suggested": self.improvements_suggested,
            "times_changed_mind": self.times_changed_mind,
            "times_defended": self.times_defended,
            "times_adopted_other": self.times_adopted_other,
            "times_won_debate": self.times_won_debate,
            "final_votes_received": self.final_votes_received,
            "total_generation_time": self.total_generation_time,
            "avg_response_time": self.avg_response_time,
        }


@dataclass
class Agent:
    id: str
    config: AgentConfig
    messages: list[AgentMessage] = field(default_factory=list)
    stats: AgentStats = field(init=False)
    
    def __post_init__(self):
        self.stats = AgentStats(
            agent_id=self.id,
            model=self.config.model,
            role=self.config.role,
        )
    
    @property
    def model(self) -> str:
        return self.config.model
    
    @property
    def role(self) -> AgentRole:
        return self.config.role
    
    @property
    def temperature(self) -> float:
        return self.config.temperature
    
    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)
        
        if message.message_type == "proposal":
            self.stats.solutions_proposed += 1
        elif message.message_type == "revision":
            self.stats.solutions_revised += 1
        elif message.message_type == "critique":
            if "bugs" in message.metadata:
                self.stats.bugs_found += len(message.metadata["bugs"])
            if "improvements" in message.metadata:
                self.stats.improvements_suggested += len(message.metadata["improvements"])
    
    def get_messages_for_round(self, round_num: int) -> list[AgentMessage]:
        return [m for m in self.messages if m.round_num == round_num]
    
    def get_latest_solution(self) -> AgentMessage | None:
        for msg in reversed(self.messages):
            if msg.message_type in ("proposal", "revision"):
                return msg
        return None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.config.model,
            "role": self.config.role.value,
            "temperature": self.config.temperature,
            "message_count": len(self.messages),
            "stats": self.stats.to_dict(),
        }
