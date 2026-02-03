"""
Tests for core functionality.
"""
import pytest
from src.core.consensus import ConsensusDetector, ConsensusConfig
from src.models import Vote, VoteType, Solution, SolutionStatus, ExecutionResult


class TestConsensusDetector:
    """Tests for ConsensusDetector."""

    @pytest.fixture
    def detector(self):
        return ConsensusDetector(ConsensusConfig(threshold=0.6, min_votes=2))

    @pytest.fixture
    def solutions(self):
        sol1 = Solution(id="sol_1", agent_id="agent_1", round_num=1, code="code1")
        sol1.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED,
            tests_passed=10,
            tests_total=10,
        )
        sol2 = Solution(id="sol_2", agent_id="agent_2", round_num=1, code="code2")
        sol2.execution_result = ExecutionResult(
            status=SolutionStatus.TEST_FAILED,
            tests_passed=5,
            tests_total=10,
        )
        return [sol1, sol2]

    def test_consensus_reached(self, detector, solutions):
        votes = [
            Vote(id="v1", agent_id="agent_1", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_1", confidence=0.9),
            Vote(id="v2", agent_id="agent_2", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_1", confidence=0.8),
            Vote(id="v3", agent_id="agent_3", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_1", confidence=0.7),
        ]
        result = detector.detect(votes, solutions, [], 1)
        assert result.reached
        assert result.winning_solution_id == "sol_1"

    def test_no_consensus(self, detector, solutions):
        votes = [
            Vote(id="v1", agent_id="agent_1", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_1", confidence=0.9),
            Vote(id="v2", agent_id="agent_2", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_2", confidence=0.8),
        ]
        result = detector.detect(votes, solutions, [], 1)
        # With 50-50 split, no consensus
        assert not result.reached

    def test_passing_bonus(self, detector, solutions):
        # With passing bonus, sol_1 (which passes tests) should have advantage
        # even with lower initial confidence
        votes = [
            Vote(id="v1", agent_id="agent_1", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_1", confidence=0.5),
            Vote(id="v2", agent_id="agent_2", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_2", confidence=0.8),
            Vote(id="v3", agent_id="agent_3", round_num=1, vote_type=VoteType.ADOPT, voted_solution_id="sol_1", confidence=0.6),
        ]
        result = detector.detect(votes, solutions, [], 1)
        # sol_1 passes all tests and has 2 votes, should win
        assert result.reached
        assert result.winning_solution_id == "sol_1"

    def test_check_perfect_solution(self, detector, solutions):
        perfect = detector.check_perfect_solution(solutions)
        assert perfect is not None
        assert perfect.id == "sol_1"

    def test_no_perfect_solution(self, detector):
        sol = Solution(id="sol_1", agent_id="agent_1", round_num=1, code="code")
        sol.execution_result = ExecutionResult(
            status=SolutionStatus.TEST_FAILED,
            tests_passed=5,
            tests_total=10,
        )
        perfect = detector.check_perfect_solution([sol])
        assert perfect is None
