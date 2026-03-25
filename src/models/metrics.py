"""
Metrics models for the LLM Code Debate System.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent import AgentStats


@dataclass
class DebateMetrics:
    """
    Comprehensive metrics for a completed debate.
    
    Used for analysis and comparison between experiments.
    """
    debate_id: str
    task_id: str
    task_difficulty: str
    
    # === Result Quality ===
    final_pass_rate: float = 0.0
    final_tests_passed: int = 0
    final_tests_total: int = 0
    improvement_over_best_initial: float = 0.0
    improvement_over_avg_initial: float = 0.0
    
    # === Debate Dynamics ===
    total_rounds: int = 0
    rounds_to_consensus: int = 0
    consensus_reached: bool = False
    consensus_ratio: float = 0.0
    
    # === Critique Stats ===
    total_critiques: int = 0
    total_bugs_found: int = 0
    total_bugs_fixed: int = 0
    bug_fix_rate: float = 0.0
    total_improvements_suggested: int = 0
    total_improvements_applied: int = 0
    
    # === Agent Behavior ===
    agent_stats: list[AgentStats] = field(default_factory=list)
    most_active_agent: str | None = None
    most_successful_agent: str | None = None
    most_bugs_found_by: str | None = None
    
    # === Code Quality ===
    initial_avg_pylint: float = 0.0
    final_pylint: float = 0.0
    initial_avg_complexity: float = 0.0
    final_complexity: float = 0.0
    
    # === Timing ===
    total_duration_seconds: float = 0.0
    avg_round_duration: float = 0.0
    total_llm_time: float = 0.0
    total_execution_time: float = 0.0
    
    # === Pass@k ===
    all_solutions_count: int = 0
    passing_solutions_count: int = 0
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0

    # === Comparison (Single-Agent Baseline) ===
    baseline_pass_rate: float | None = None
    improvement_over_baseline: float | None = None
    
    def compute_derived_metrics(self) -> None:
        """Compute derived metrics from raw data."""
        # Bug fix rate
        if self.total_bugs_found > 0:
            self.bug_fix_rate = self.total_bugs_fixed / self.total_bugs_found
        
        # Average round duration
        if self.total_rounds > 0:
            self.avg_round_duration = self.total_duration_seconds / self.total_rounds
        
        # Find most active/successful agents
        if self.agent_stats:
            # Most active = most critiques given
            most_active = max(self.agent_stats, key=lambda s: s.critiques_given)
            self.most_active_agent = most_active.agent_id
            
            # Most successful = won debate
            winners = [s for s in self.agent_stats if s.times_won_debate > 0]
            if winners:
                self.most_successful_agent = winners[0].agent_id
            
            # Most bugs found
            most_bugs = max(self.agent_stats, key=lambda s: s.bugs_found)
            if most_bugs.bugs_found > 0:
                self.most_bugs_found_by = most_bugs.agent_id
        
        # Improvement over baseline
        if self.baseline_pass_rate is not None:
            if self.baseline_pass_rate > 0:
                self.improvement_over_baseline = (
                    (self.final_pass_rate - self.baseline_pass_rate) / self.baseline_pass_rate
                )
            elif self.final_pass_rate > 0:
                self.improvement_over_baseline = float('inf')
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "task_id": self.task_id,
            "task_difficulty": self.task_difficulty,
            "final_pass_rate": self.final_pass_rate,
            "final_tests_passed": self.final_tests_passed,
            "final_tests_total": self.final_tests_total,
            "improvement_over_best_initial": self.improvement_over_best_initial,
            "improvement_over_avg_initial": self.improvement_over_avg_initial,
            "total_rounds": self.total_rounds,
            "rounds_to_consensus": self.rounds_to_consensus,
            "consensus_reached": self.consensus_reached,
            "consensus_ratio": self.consensus_ratio,
            "total_critiques": self.total_critiques,
            "total_bugs_found": self.total_bugs_found,
            "total_bugs_fixed": self.total_bugs_fixed,
            "bug_fix_rate": self.bug_fix_rate,
            "total_improvements_suggested": self.total_improvements_suggested,
            "agent_stats": [s.to_dict() for s in self.agent_stats],
            "most_active_agent": self.most_active_agent,
            "most_successful_agent": self.most_successful_agent,
            "initial_avg_pylint": self.initial_avg_pylint,
            "final_pylint": self.final_pylint,
            "initial_avg_complexity": self.initial_avg_complexity,
            "final_complexity": self.final_complexity,
            "total_duration_seconds": self.total_duration_seconds,
            "avg_round_duration": self.avg_round_duration,
            "all_solutions_count": self.all_solutions_count,
            "passing_solutions_count": self.passing_solutions_count,
            "pass_at_1": self.pass_at_1,
            "pass_at_3": self.pass_at_3,
            "baseline_pass_rate": self.baseline_pass_rate,
            "improvement_over_baseline": self.improvement_over_baseline,
        }


@dataclass
class ExperimentSummary:
    """
    Summary of multiple debates for an experiment.
    """
    experiment_id: str
    experiment_name: str
    
    # Configuration
    num_agents: int = 0
    agent_models: list[str] = field(default_factory=list)
    max_rounds: int = 5
    
    # Task stats
    total_tasks: int = 0
    tasks_by_difficulty: dict[str, int] = field(default_factory=dict)
    
    # Results
    overall_pass_rate: float = 0.0
    pass_rate_by_difficulty: dict[str, float] = field(default_factory=dict)
    avg_rounds_to_consensus: float = 0.0
    consensus_rate: float = 0.0
    
    # Pass@k
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0
    pass_at_5: float = 0.0
    pass_at_k_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)

    # Comparisons
    vs_single_agent_improvement: float = 0.0
    vs_best_single_model: dict[str, float] = field(default_factory=dict)
    
    # Timing
    total_duration: float = 0.0
    avg_debate_duration: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "num_agents": self.num_agents,
            "agent_models": self.agent_models,
            "max_rounds": self.max_rounds,
            "total_tasks": self.total_tasks,
            "tasks_by_difficulty": self.tasks_by_difficulty,
            "overall_pass_rate": self.overall_pass_rate,
            "pass_rate_by_difficulty": self.pass_rate_by_difficulty,
            "avg_rounds_to_consensus": self.avg_rounds_to_consensus,
            "consensus_rate": self.consensus_rate,
            "pass_at_1": self.pass_at_1,
            "pass_at_3": self.pass_at_3,
            "pass_at_5": self.pass_at_5,
            "pass_at_k_by_difficulty": self.pass_at_k_by_difficulty,
            "vs_single_agent_improvement": self.vs_single_agent_improvement,
            "total_duration": self.total_duration,
            "avg_debate_duration": self.avg_debate_duration,
        }


@dataclass
class AgentProfile:
    """
    Aggregated profile of an agent's behavior across multiple debates.
    """
    model: str
    total_debates: int = 0
    
    # Win rate
    debates_won: int = 0
    win_rate: float = 0.0
    
    # Behavior patterns
    avg_critiques_per_debate: float = 0.0
    avg_bugs_found_per_debate: float = 0.0
    times_changed_mind_ratio: float = 0.0
    times_defended_ratio: float = 0.0
    
    # Quality contribution
    avg_solution_pass_rate: float = 0.0
    avg_solution_quality: float = 0.0
    
    # Personality classification
    personality_type: str = "balanced"  # "aggressive_critic", "passive_adopter", "stubborn_defender", etc.
    
    def classify_personality(self) -> None:
        """Classify agent personality based on behavior patterns."""
        if self.avg_bugs_found_per_debate > 2.0 and self.times_changed_mind_ratio < 0.3:
            self.personality_type = "aggressive_critic"
        elif self.times_changed_mind_ratio > 0.6:
            self.personality_type = "passive_adopter"
        elif self.times_defended_ratio > 0.7:
            self.personality_type = "stubborn_defender"
        elif self.avg_critiques_per_debate > 3.0:
            self.personality_type = "active_collaborator"
        else:
            self.personality_type = "balanced"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_debates": self.total_debates,
            "debates_won": self.debates_won,
            "win_rate": self.win_rate,
            "avg_critiques_per_debate": self.avg_critiques_per_debate,
            "avg_bugs_found_per_debate": self.avg_bugs_found_per_debate,
            "times_changed_mind_ratio": self.times_changed_mind_ratio,
            "times_defended_ratio": self.times_defended_ratio,
            "avg_solution_pass_rate": self.avg_solution_pass_rate,
            "personality_type": self.personality_type,
        }
