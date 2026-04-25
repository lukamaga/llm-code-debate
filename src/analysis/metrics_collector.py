"""
Metrics collection and analysis for debates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from typing import Any

from ..models import (
    AgentProfile,
    AgentStats,
    Debate,
    DebateMetrics,
    ExperimentSummary,
    Solution,
)
from ..models.solution import SolutionStatus


class MetricsCollector:
    """
    Collects and computes metrics from debates.
    """
    
    def collect_debate_metrics(self, debate: Debate) -> DebateMetrics:
        """
        Collect comprehensive metrics from a completed debate.
        
        Args:
            debate: The completed debate.
            
        Returns:
            DebateMetrics with all computed metrics.
        """
        metrics = DebateMetrics(
            debate_id=debate.id,
            task_id=debate.task.id,
            task_difficulty=debate.task.difficulty,
        )
        
        # Result quality
        if debate.final_solution:
            metrics.final_pass_rate = debate.final_solution.pass_rate
            if debate.final_solution.execution_result:
                metrics.final_tests_passed = debate.final_solution.execution_result.tests_passed
                metrics.final_tests_total = debate.final_solution.execution_result.tests_total
        
        # Pass@k: count all solutions across rounds
        n, c = 0, 0
        for round_data in debate.rounds:
            for sol in round_data.solutions:
                n += 1
                if sol.execution_result and sol.execution_result.all_passed:
                    c += 1
        metrics.all_solutions_count = n
        metrics.passing_solutions_count = c
        if n > 0:
            metrics.pass_at_1 = compute_pass_at_k(n, c, min(1, n))
            metrics.pass_at_3 = compute_pass_at_k(n, c, min(3, n))

        # Improvement over initial
        if debate.rounds:
            initial_solutions = debate.rounds[0].solutions
            if initial_solutions:
                initial_pass_rates = [s.pass_rate for s in initial_solutions]
                best_initial = max(initial_pass_rates)
                avg_initial = sum(initial_pass_rates) / len(initial_pass_rates)
                
                if best_initial > 0:
                    metrics.improvement_over_best_initial = (
                        (metrics.final_pass_rate - best_initial) / best_initial
                    )
                elif metrics.final_pass_rate > 0:
                    # initial was 0% but we improved — use absolute difference
                    metrics.improvement_over_best_initial = metrics.final_pass_rate

                if avg_initial > 0:
                    metrics.improvement_over_avg_initial = (
                        (metrics.final_pass_rate - avg_initial) / avg_initial
                    )
                elif metrics.final_pass_rate > 0:
                    metrics.improvement_over_avg_initial = metrics.final_pass_rate
        
        # Debate dynamics
        metrics.total_rounds = debate.total_rounds
        metrics.consensus_reached = debate.final_consensus.reached if debate.final_consensus else False
        metrics.consensus_ratio = debate.final_consensus.consensus_ratio if debate.final_consensus else 0.0
        metrics.rounds_to_consensus = (
            debate.final_consensus.round_num
            if metrics.consensus_reached and debate.final_consensus
            else 0
        )
        
        # Critique stats
        all_critiques = debate.all_critiques
        metrics.total_critiques = len(all_critiques)
        metrics.total_bugs_found = sum(len(c.bugs) for c in all_critiques)
        metrics.total_improvements_suggested = sum(len(c.improvements) for c in all_critiques)
        
        # Count bugs fixed (compare first and last round pass rates per agent)
        # This is an approximation
        if len(debate.rounds) >= 2:
            first_round = debate.rounds[0]
            last_round = debate.rounds[-1]
            bugs_fixed = 0
            for sol1 in first_round.solutions:
                sol2 = next(
                    (s for s in last_round.solutions if s.agent_id == sol1.agent_id),
                    None
                )
                if sol1.execution_result and sol2 and sol2.execution_result:
                    improvement = (
                        sol2.execution_result.tests_passed - 
                        sol1.execution_result.tests_passed
                    )
                    if improvement > 0:
                        bugs_fixed += improvement
            metrics.total_bugs_fixed = bugs_fixed
        
        # Agent stats
        metrics.agent_stats = [agent.stats for agent in debate.agents]
        
        # Code quality
        initial_pylints = []
        for sol in debate.rounds[0].solutions if debate.rounds else []:
            if sol.quality_metrics:
                initial_pylints.append(sol.quality_metrics.pylint_score)
        if initial_pylints:
            metrics.initial_avg_pylint = sum(initial_pylints) / len(initial_pylints)
        
        if debate.final_solution and debate.final_solution.quality_metrics:
            metrics.final_pylint = debate.final_solution.quality_metrics.pylint_score
            metrics.final_complexity = debate.final_solution.quality_metrics.cyclomatic_complexity
        
        # Timing
        metrics.total_duration_seconds = debate.duration_seconds
        metrics.total_llm_time = sum(
            agent.stats.total_generation_time for agent in debate.agents
        )
        
        # Compute derived metrics
        metrics.compute_derived_metrics()
        
        return metrics
    
    def collect_experiment_summary(
        self,
        experiment_id: str,
        experiment_name: str,
        debates: list[Debate],
    ) -> ExperimentSummary:
        """
        Collect summary metrics for an experiment.
        
        Args:
            experiment_id: ID of the experiment.
            experiment_name: Name of the experiment.
            debates: List of debates in the experiment.
            
        Returns:
            ExperimentSummary with aggregated metrics.
        """
        summary = ExperimentSummary(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
        )
        
        if not debates:
            return summary
        
        # Configuration (from first debate)
        first = debates[0]
        summary.num_agents = len(first.agents)
        summary.agent_models = list(set(a.model for a in first.agents))
        summary.max_rounds = first.max_rounds
        
        # Task stats
        summary.total_tasks = len(debates)
        by_difficulty: dict[str, int] = defaultdict(int)
        for d in debates:
            by_difficulty[d.task.difficulty] += 1
        summary.tasks_by_difficulty = dict(by_difficulty)
        
        # Results
        pass_rates = []
        rounds_list = []
        consensus_count = 0
        
        pass_by_diff: dict[str, list[float]] = defaultdict(list)
        
        for debate in debates:
            if debate.final_solution:
                pr = debate.final_solution.pass_rate
                pass_rates.append(pr)
                pass_by_diff[debate.task.difficulty].append(pr)
            
            rounds_list.append(debate.total_rounds)
            
            if debate.final_consensus and debate.final_consensus.reached:
                consensus_count += 1
        
        summary.overall_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        summary.avg_rounds_to_consensus = sum(rounds_list) / len(rounds_list) if rounds_list else 0.0
        summary.consensus_rate = consensus_count / len(debates) if debates else 0.0
        
        # Pass rate by difficulty
        for diff, rates in pass_by_diff.items():
            summary.pass_rate_by_difficulty[diff] = sum(rates) / len(rates)

        # Pass@k computation
        debates_by_task: dict[str, list[Debate]] = defaultdict(list)
        for debate in debates:
            debates_by_task[debate.task.id].append(debate)

        total_pass_1: list[float] = []
        total_pass_3: list[float] = []
        total_pass_5: list[float] = []
        pass_k_by_diff: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for task_id, task_debates in debates_by_task.items():
            n = 0
            c = 0
            difficulty = task_debates[0].task.difficulty

            for td in task_debates:
                for round_data in td.rounds:
                    for sol in round_data.solutions:
                        n += 1
                        if sol.execution_result and sol.execution_result.all_passed:
                            c += 1

            if n > 0:
                p1 = compute_pass_at_k(n, c, min(1, n))
                p3 = compute_pass_at_k(n, c, min(3, n))
                p5 = compute_pass_at_k(n, c, min(5, n))
                total_pass_1.append(p1)
                total_pass_3.append(p3)
                total_pass_5.append(p5)
                pass_k_by_diff[difficulty]["pass@1"].append(p1)
                pass_k_by_diff[difficulty]["pass@3"].append(p3)
                pass_k_by_diff[difficulty]["pass@5"].append(p5)

        summary.pass_at_1 = sum(total_pass_1) / len(total_pass_1) if total_pass_1 else 0.0
        summary.pass_at_3 = sum(total_pass_3) / len(total_pass_3) if total_pass_3 else 0.0
        summary.pass_at_5 = sum(total_pass_5) / len(total_pass_5) if total_pass_5 else 0.0

        for diff, metrics_dict in pass_k_by_diff.items():
            summary.pass_at_k_by_difficulty[diff] = {
                k_name: sum(v) / len(v) for k_name, v in metrics_dict.items()
            }

        # Timing
        summary.total_duration = sum(d.duration_seconds for d in debates)
        summary.avg_debate_duration = summary.total_duration / len(debates)
        
        return summary
    
    def build_agent_profile(
        self,
        model: str,
        stats_list: list[AgentStats],
    ) -> AgentProfile:
        """
        Build an aggregated profile for an agent model.
        
        Args:
            model: The model name.
            stats_list: List of stats from different debates.
            
        Returns:
            AgentProfile with aggregated behavior patterns.
        """
        profile = AgentProfile(model=model)
        
        if not stats_list:
            return profile
        
        profile.total_debates = len(stats_list)
        
        # Win rate
        profile.debates_won = sum(1 for s in stats_list if s.times_won_debate > 0)
        profile.win_rate = profile.debates_won / profile.total_debates
        
        # Averages
        profile.avg_critiques_per_debate = (
            sum(s.critiques_given for s in stats_list) / profile.total_debates
        )
        profile.avg_bugs_found_per_debate = (
            sum(s.bugs_found for s in stats_list) / profile.total_debates
        )
        
        # Behavior ratios
        total_decisions = sum(
            s.times_changed_mind + s.times_defended for s in stats_list
        )
        if total_decisions > 0:
            profile.times_changed_mind_ratio = (
                sum(s.times_changed_mind for s in stats_list) / total_decisions
            )
            profile.times_defended_ratio = (
                sum(s.times_defended for s in stats_list) / total_decisions
            )
        
        # Classify personality
        profile.classify_personality()
        
        return profile


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric (Chen et al., 2021).

    Calculates the probability that at least one of k randomly selected
    solutions from n total solutions passes all tests, given that c solutions
    pass all tests.

    Args:
        n: Total number of generated solutions.
        c: Number of solutions that pass all tests.
        k: Number of attempts (samples).

    Returns:
        Probability that at least 1 of k solutions passes all tests.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n or n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compare_single_vs_multi(
    single_agent_results: list[dict[str, Any]],
    multi_agent_results: list[DebateMetrics],
) -> dict[str, Any]:
    """
    Compare single-agent vs multi-agent performance.
    
    Args:
        single_agent_results: Results from single-agent runs.
        multi_agent_results: Metrics from multi-agent debates.
        
    Returns:
        Comparison statistics.
    """
    # Group by task
    single_by_task = {r["task_id"]: r for r in single_agent_results}
    multi_by_task = {m.task_id: m for m in multi_agent_results}
    
    common_tasks = set(single_by_task.keys()) & set(multi_by_task.keys())
    
    improvements = []
    for task_id in common_tasks:
        single_pr = single_by_task[task_id].get("pass_rate", 0)
        multi_pr = multi_by_task[task_id].final_pass_rate
        
        if single_pr > 0:
            improvement = (multi_pr - single_pr) / single_pr
        elif multi_pr > 0:
            improvement = float('inf')
        else:
            improvement = 0
        
        improvements.append({
            "task_id": task_id,
            "single_pass_rate": single_pr,
            "multi_pass_rate": multi_pr,
            "improvement": improvement,
        })
    
    valid_improvements = [i["improvement"] for i in improvements if i["improvement"] != float('inf')]
    
    return {
        "total_tasks_compared": len(common_tasks),
        "avg_improvement": sum(valid_improvements) / len(valid_improvements) if valid_improvements else 0,
        "tasks_improved": sum(1 for i in improvements if i["improvement"] > 0),
        "tasks_worse": sum(1 for i in improvements if i["improvement"] < 0),
        "tasks_same": sum(1 for i in improvements if i["improvement"] == 0),
        "details": improvements,
    }
