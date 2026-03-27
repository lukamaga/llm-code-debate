"""
Database models for persisting debate results.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class DebateRecord(Base):
    """Record of a completed debate."""
    
    __tablename__ = "debates"
    
    id = Column(String(50), primary_key=True)
    task_id = Column(String(100), nullable=False, index=True)
    task_name = Column(String(200))
    task_difficulty = Column(String(50), index=True)
    
    # Status
    status = Column(String(50), nullable=False)
    error_message = Column(Text)
    
    # Configuration
    num_agents = Column(Integer)
    max_rounds = Column(Integer)
    consensus_threshold = Column(Float)
    agent_models = Column(JSON)  # List of model names
    
    # Results
    final_pass_rate = Column(Float)
    tests_passed = Column(Integer)
    tests_total = Column(Integer)
    winning_agent_id = Column(String(100))
    consensus_reached = Column(Boolean, default=False)
    consensus_ratio = Column(Float)
    
    # Timing
    total_rounds = Column(Integer)
    duration_seconds = Column(Float)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)

    # Mode
    is_solo = Column(Boolean, default=False)

    # Full data (JSON)
    full_debate_data = Column(JSON)
    
    # Relationships
    rounds = relationship("RoundRecord", back_populates="debate", cascade="all, delete-orphan")
    agent_stats = relationship("AgentStatRecord", back_populates="debate", cascade="all, delete-orphan")
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_difficulty": self.task_difficulty,
            "status": self.status,
            "num_agents": self.num_agents,
            "final_pass_rate": self.final_pass_rate,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "winning_agent_id": self.winning_agent_id,
            "consensus_reached": self.consensus_reached,
            "total_rounds": self.total_rounds,
            "duration_seconds": self.duration_seconds,
            "is_solo": self.is_solo or False,
        }


class RoundRecord(Base):
    """Record of a single debate round."""
    
    __tablename__ = "rounds"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    debate_id = Column(String(50), ForeignKey("debates.id"), nullable=False)
    round_num = Column(Integer, nullable=False)
    
    # Stats
    best_pass_rate = Column(Float)
    avg_pass_rate = Column(Float)
    bugs_found = Column(Integer)
    improvements_suggested = Column(Integer)
    
    # Timing
    duration_seconds = Column(Float)
    
    # Data
    solutions_data = Column(JSON)
    critiques_data = Column(JSON)
    votes_data = Column(JSON)
    consensus_data = Column(JSON)
    
    # Relationship
    debate = relationship("DebateRecord", back_populates="rounds")


class AgentStatRecord(Base):
    """Record of agent statistics for a debate."""
    
    __tablename__ = "agent_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    debate_id = Column(String(50), ForeignKey("debates.id"), nullable=False)
    agent_id = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False, index=True)
    role = Column(String(50))
    
    # Stats
    solutions_proposed = Column(Integer, default=0)
    solutions_revised = Column(Integer, default=0)
    critiques_given = Column(Integer, default=0)
    bugs_found = Column(Integer, default=0)
    times_changed_mind = Column(Integer, default=0)
    times_defended = Column(Integer, default=0)
    times_won_debate = Column(Integer, default=0)
    total_generation_time = Column(Float, default=0.0)
    
    # Relationship
    debate = relationship("DebateRecord", back_populates="agent_stats")


class TaskRecord(Base):
    """Record of a task."""
    
    __tablename__ = "tasks"
    
    id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=False)
    difficulty = Column(String(50), index=True)
    description = Column(Text)
    signature = Column(Text)
    tests = Column(JSON)
    constraints = Column(JSON)
    tags = Column(JSON)
    
    # Stats
    total_debates = Column(Integer, default=0)
    avg_pass_rate = Column(Float)
    best_pass_rate = Column(Float)


class ExperimentRecord(Base):
    """Record of an experiment (multiple debates)."""
    
    __tablename__ = "experiments"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Configuration
    config = Column(JSON)
    
    # Results
    total_debates = Column(Integer, default=0)
    overall_pass_rate = Column(Float)
    avg_rounds = Column(Float)
    consensus_rate = Column(Float)
    
    # Timing
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    # Debate IDs
    debate_ids = Column(JSON)


def create_database(db_path: str = "debate_results.db") -> sessionmaker:
    """
    Create database and return session maker.
    
    Args:
        db_path: Path to SQLite database file.
        
    Returns:
        sessionmaker for creating database sessions.
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
