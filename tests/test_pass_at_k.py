import pytest
from math import comb

from src.analysis.metrics_collector import compute_pass_at_k, MetricsCollector
from src.models import (
    Agent,
    AgentConfig,
    ConsensusResult,
    Debate,
    DebateStatus,
    ExecutionResult,
    RoundSummary,
    Solution,
    SolutionStatus,
    Task,
)


class TestComputePassAtK:

    def test_all_pass(self):
        assert compute_pass_at_k(10, 10, 1) == 1.0
        assert compute_pass_at_k(10, 10, 5) == 1.0

    def test_none_pass(self):
        assert compute_pass_at_k(10, 0, 1) == 0.0
        assert compute_pass_at_k(10, 0, 5) == 0.0

    def test_pass_at_1_partial(self):
        result = compute_pass_at_k(10, 3, 1)
        assert result == pytest.approx(0.3)

    def test_pass_at_k_increases_with_k(self):
        p1 = compute_pass_at_k(10, 3, 1)
        p3 = compute_pass_at_k(10, 3, 3)
        p5 = compute_pass_at_k(10, 3, 5)
        assert p1 < p3 < p5

    def test_pass_at_k_formula(self):
        expected = 1.0 - comb(7, 3) / comb(10, 3)
        assert compute_pass_at_k(10, 3, 3) == pytest.approx(expected)

    def test_k_greater_than_failures(self):
        assert compute_pass_at_k(10, 8, 5) == 1.0

    def test_edge_case_n_zero(self):
        assert compute_pass_at_k(0, 0, 1) == 0.0

    def test_edge_case_k_zero(self):
        assert compute_pass_at_k(10, 5, 0) == 0.0

    def test_single_solution_passing(self):
        assert compute_pass_at_k(1, 1, 1) == 1.0

    def test_single_solution_failing(self):
        assert compute_pass_at_k(1, 0, 1) == 0.0


def _make_task(task_id="task_1", difficulty="easy"):
    return Task(
        id=task_id,
        name=f"Task {task_id}",
        description="Test task",
        difficulty=difficulty,
        signature="def solve(): pass",
        tests=["def test_1(): assert True"],
    )


def _make_solution(agent_id, round_num, passed, total):
    sol = Solution(
        id=f"sol_{agent_id}_r{round_num}",
        agent_id=agent_id,
        round_num=round_num,
        code="def solve(): pass",
    )
    sol.execution_result = ExecutionResult(
        status=SolutionStatus.PASSED if passed == total else SolutionStatus.TEST_FAILED,
        tests_passed=passed,
        tests_total=total,
    )
    return sol


def _make_debate(task, rounds_data, status=DebateStatus.CONSENSUS_REACHED):
    agents = []
    seen = set()
    rounds = []
    for round_num, solutions_data in enumerate(rounds_data, 1):
        solutions = []
        for agent_id, passed, total in solutions_data:
            sol = _make_solution(agent_id, round_num, passed, total)
            solutions.append(sol)
            if agent_id not in seen:
                seen.add(agent_id)
                agents.append(Agent(
                    id=agent_id,
                    config=AgentConfig(name=agent_id, model="test-model"),
                ))
        rounds.append(RoundSummary(
            round_num=round_num,
            solutions=solutions,
            critiques=[],
            votes=[],
        ))

    debate = Debate(
        id=f"debate_{task.id}",
        task=task,
        agents=agents,
        max_rounds=5,
        status=status,
    )
    debate.rounds = rounds
    if rounds and rounds[-1].solutions:
        debate.final_solution = rounds[-1].solutions[0]
    debate.final_consensus = ConsensusResult(reached=True, consensus_ratio=1.0)
    return debate


class TestDebateMetricsPassAtK:

    def test_debate_metrics_all_pass(self):
        task = _make_task()
        debate = _make_debate(task, [
            [("a1", 3, 3), ("a2", 3, 3)],
            [("a1", 3, 3), ("a2", 3, 3)],
        ])
        collector = MetricsCollector()
        metrics = collector.collect_debate_metrics(debate)
        assert metrics.all_solutions_count == 4
        assert metrics.passing_solutions_count == 4
        assert metrics.pass_at_1 == 1.0

    def test_debate_metrics_partial_pass(self):
        task = _make_task()
        debate = _make_debate(task, [
            [("a1", 3, 3), ("a2", 1, 3)],
        ])
        collector = MetricsCollector()
        metrics = collector.collect_debate_metrics(debate)
        assert metrics.all_solutions_count == 2
        assert metrics.passing_solutions_count == 1
        assert metrics.pass_at_1 == pytest.approx(0.5)

    def test_debate_metrics_none_pass(self):
        task = _make_task()
        debate = _make_debate(task, [
            [("a1", 0, 3), ("a2", 1, 3)],
        ])
        collector = MetricsCollector()
        metrics = collector.collect_debate_metrics(debate)
        assert metrics.passing_solutions_count == 0
        assert metrics.pass_at_1 == 0.0

    def test_debate_metrics_to_dict(self):
        task = _make_task()
        debate = _make_debate(task, [
            [("a1", 3, 3), ("a2", 1, 3)],
        ])
        collector = MetricsCollector()
        metrics = collector.collect_debate_metrics(debate)
        d = metrics.to_dict()
        assert "pass_at_1" in d
        assert "pass_at_3" in d
        assert "all_solutions_count" in d
        assert "passing_solutions_count" in d


class TestExperimentSummaryPassAtK:

    def test_experiment_pass_at_k(self):
        task1 = _make_task("t1", "easy")
        task2 = _make_task("t2", "hard")
        debate1 = _make_debate(task1, [
            [("a1", 3, 3), ("a2", 3, 3)],
        ])
        debate2 = _make_debate(task2, [
            [("a1", 0, 3), ("a2", 0, 3)],
        ])
        collector = MetricsCollector()
        summary = collector.collect_experiment_summary("exp1", "test", [debate1, debate2])

        assert summary.pass_at_1 == pytest.approx(0.5)

    def test_experiment_pass_at_k_by_difficulty(self):
        task1 = _make_task("t1", "easy")
        task2 = _make_task("t2", "easy")
        debate1 = _make_debate(task1, [
            [("a1", 3, 3), ("a2", 0, 3)],
        ])
        debate2 = _make_debate(task2, [
            [("a1", 3, 3), ("a2", 3, 3)],
        ])
        collector = MetricsCollector()
        summary = collector.collect_experiment_summary("exp1", "test", [debate1, debate2])

        assert "easy" in summary.pass_at_k_by_difficulty
        easy_metrics = summary.pass_at_k_by_difficulty["easy"]
        assert "pass@1" in easy_metrics
        assert easy_metrics["pass@1"] == pytest.approx(0.75)

    def test_experiment_summary_to_dict(self):
        task = _make_task()
        debate = _make_debate(task, [
            [("a1", 3, 3)],
        ])
        collector = MetricsCollector()
        summary = collector.collect_experiment_summary("exp1", "test", [debate])
        d = summary.to_dict()
        assert "pass_at_1" in d
        assert "pass_at_3" in d
        assert "pass_at_5" in d
        assert "pass_at_k_by_difficulty" in d
