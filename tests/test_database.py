"""
Tests for DebateRepository with in-memory SQLite.

Note: The repository closes sessions after each operation, so returned ORM
objects are detached. Tests that need to inspect relationships (rounds,
agent_stats) must re-query within a session.
"""
from __future__ import annotations

import pytest

from src.database.repository import DebateRepository
from src.database.models import DebateRecord, TaskRecord, ExperimentRecord
from src.models import (
    Agent,
    AgentConfig,
    ConsensusResult,
    Debate,
    DebateConfig,
    DebateStatus,
    ExecutionResult,
    RoundSummary,
    Solution,
    SolutionStatus,
    Task,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_task(task_id="t1", difficulty="easy"):
    return Task(
        id=task_id, name=f"Task {task_id}", difficulty=difficulty,
        description="desc", signature="def f():", tests=["assert f() == 1"],
    )


def _make_completed_debate(
    debate_id="d1", task_id="t1", difficulty="easy",
    status=DebateStatus.CONSENSUS_REACHED,
    tests_passed=3, tests_total=3,
):
    task = _make_task(task_id, difficulty)
    agents = [
        Agent(id="a1", config=AgentConfig(name="a1", model="m1:7b")),
        Agent(id="a2", config=AgentConfig(name="a2", model="m2:7b")),
    ]
    debate = Debate(id=debate_id, task=task, agents=agents, max_rounds=3)

    sol_status = SolutionStatus.PASSED if tests_passed == tests_total else SolutionStatus.TEST_FAILED
    sol = Solution(id="sol1", agent_id="a1", round_num=1, code="def f(): return 1")
    sol.execution_result = ExecutionResult(
        status=sol_status,
        tests_passed=tests_passed,
        tests_total=tests_total,
    )
    rs = RoundSummary(round_num=1, solutions=[sol], critiques=[], votes=[])
    debate.add_round(rs)

    consensus = ConsensusResult(
        reached=(status == DebateStatus.CONSENSUS_REACHED),
        winning_solution_id=sol.id,
        winning_agent_id=sol.agent_id,
        consensus_ratio=0.9,
        round_num=1,
    ) if status == DebateStatus.CONSENSUS_REACHED else None

    debate.finalize(status=status, final_solution=sol, consensus=consensus)
    return debate


def _query_debate(repo, debate_id):
    """Query a debate record within a session to avoid DetachedInstanceError."""
    session = repo._get_session()
    try:
        record = session.query(DebateRecord).filter_by(id=debate_id).first()
        if record is None:
            return None
        # Eagerly access relationships while session is open
        result = {
            "id": record.id,
            "status": record.status,
            "task_id": record.task_id,
            "num_rounds": len(record.rounds),
            "num_agent_stats": len(record.agent_stats),
            "final_pass_rate": record.final_pass_rate,
            "consensus_reached": record.consensus_reached,
            "total_rounds": record.total_rounds,
        }
        return result
    finally:
        session.close()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def repo():
    """DebateRepository backed by in-memory SQLite."""
    return DebateRepository(db_path=":memory:")


# =============================================================================
# TestDebateRepository
# =============================================================================

class TestDebateRepository:
    def test_save_and_get_debate(self, repo):
        debate = _make_completed_debate("d1")
        repo.save_debate(debate)
        result = _query_debate(repo, "d1")
        assert result is not None
        assert result["id"] == "d1"
        assert result["status"] == "consensus_reached"

    def test_save_debate_with_rounds(self, repo):
        debate = _make_completed_debate("d2")
        repo.save_debate(debate)
        result = _query_debate(repo, "d2")
        assert result["num_rounds"] == 1

    def test_save_debate_with_agent_stats(self, repo):
        debate = _make_completed_debate("d3")
        repo.save_debate(debate)
        result = _query_debate(repo, "d3")
        assert result["num_agent_stats"] == 2

    def test_get_nonexistent_debate(self, repo):
        assert repo.get_debate("nope") is None

    def test_get_debates_by_task(self, repo):
        repo.save_debate(_make_completed_debate("d1", task_id="t1"))
        repo.save_debate(_make_completed_debate("d2", task_id="t2"))
        repo.save_debate(_make_completed_debate("d3", task_id="t1"))
        results = repo.get_debates_by_task("t1")
        assert len(results) == 2

    def test_get_debates_by_difficulty(self, repo):
        repo.save_debate(_make_completed_debate("d1", difficulty="easy"))
        repo.save_debate(_make_completed_debate("d2", difficulty="hard"))
        results = repo.get_debates_by_difficulty("easy")
        assert len(results) == 1

    def test_get_all_debates(self, repo):
        repo.save_debate(_make_completed_debate("d1"))
        repo.save_debate(_make_completed_debate("d2"))
        results = repo.get_all_debates()
        assert len(results) == 2

    def test_get_all_debates_limit(self, repo):
        for i in range(5):
            repo.save_debate(_make_completed_debate(f"d{i}"))
        results = repo.get_all_debates(limit=3)
        assert len(results) == 3


class TestTaskRepository:
    def test_save_task(self, repo):
        task = _make_task("t1")
        repo.save_task(task)
        # Re-query to verify
        session = repo._get_session()
        try:
            record = session.query(TaskRecord).filter_by(id="t1").first()
            assert record is not None
            assert record.name == "Task t1"
        finally:
            session.close()

    def test_save_task_idempotent(self, repo):
        task = _make_task("t1")
        repo.save_task(task)
        repo.save_task(task)
        # Should still be only one record
        session = repo._get_session()
        try:
            count = session.query(TaskRecord).count()
            assert count == 1
        finally:
            session.close()


class TestSummaryStats:
    def test_empty(self, repo):
        stats = repo.get_summary_stats()
        assert stats["total_debates"] == 0
        assert stats["overall_pass_rate"] == 0.0

    def test_with_data(self, repo):
        # d1: 3/3 passed (1.0), d2: 1/3 passed (~0.333)
        repo.save_debate(_make_completed_debate("d1", tests_passed=3, tests_total=3))
        repo.save_debate(_make_completed_debate(
            "d2", tests_passed=1, tests_total=3,
            status=DebateStatus.MAX_ROUNDS_REACHED,
        ))
        stats = repo.get_summary_stats()
        assert stats["total_debates"] == 2
        assert stats["consensus_rate"] == 0.5
        # avg pass_rate = (1.0 + 0.333) / 2 ≈ 0.667
        assert stats["overall_pass_rate"] == pytest.approx(0.667, abs=0.01)


class TestExperiment:
    def test_create_experiment(self, repo):
        repo.create_experiment("exp1", "Test Experiment", "desc")
        session = repo._get_session()
        try:
            exp = session.query(ExperimentRecord).filter_by(id="exp1").first()
            assert exp is not None
            assert exp.name == "Test Experiment"
        finally:
            session.close()

    def test_add_debate_to_experiment(self, repo):
        repo.create_experiment("exp1", "Test")
        repo.add_debate_to_experiment("exp1", "d1")
        repo.add_debate_to_experiment("exp1", "d2")
        session = repo._get_session()
        try:
            exp = session.query(ExperimentRecord).filter_by(id="exp1").first()
            assert exp.total_debates >= 1
            assert "d1" in exp.debate_ids
        finally:
            session.close()
