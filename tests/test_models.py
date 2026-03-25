"""
Tests for data models: Task, Solution, Agent, Critique, Debate, DebateConfig, Metrics.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.models import (
    Agent,
    AgentConfig,
    AgentMessage,
    AgentProfile,
    AgentRole,
    AgentStats,
    Bug,
    BugSeverity,
    CodeQualityMetrics,
    ConsensusResult,
    Critique,
    Debate,
    DebateConfig,
    DebateMetrics,
    DebateStatus,
    ExecutionResult,
    Improvement,
    ImprovementType,
    RoundSummary,
    Solution,
    SolutionStatus,
    Task,
    Vote,
    VoteType,
)


# =============================================================================
# TestTask
# =============================================================================

class TestTask:
    def test_from_dict(self):
        data = {
            "id": "t1", "name": "Two Sum", "difficulty": "easy",
            "description": "Find two numbers", "signature": "def two_sum(nums, target):",
            "tests": ["assert two_sum([2,7], 9) == [0,1]"],
            "constraints": ["1 <= len(nums) <= 100"],
            "tags": ["array", "hash"],
        }
        task = Task.from_dict(data)
        assert task.id == "t1"
        assert task.difficulty == "easy"
        assert task.tags == ["array", "hash"]

    def test_from_dict_defaults(self):
        data = {
            "id": "t2", "name": "Min", "description": "Find min",
            "signature": "def find_min(arr):", "tests": ["assert find_min([1]) == 1"],
        }
        task = Task.from_dict(data)
        assert task.difficulty == "medium"
        assert task.constraints == []
        assert task.tags == []

    def test_to_dict_roundtrip(self, sample_task):
        d = sample_task.to_dict()
        restored = Task.from_dict(d)
        assert restored.id == sample_task.id
        assert restored.tests == sample_task.tests

    def test_get_test_code(self, sample_task):
        code = sample_task.get_test_code()
        assert "test_basic" in code


# =============================================================================
# TestSolution
# =============================================================================

class TestSolution:
    def test_extract_code_block_python(self):
        sol = Solution(id="s", agent_id="a", round_num=1,
                       code="```python\ndef f(): pass\n```")
        assert sol.extract_code_block() == "def f(): pass"

    def test_extract_code_block_generic(self):
        sol = Solution(id="s", agent_id="a", round_num=1,
                       code="```\ndef f(): pass\n```")
        assert sol.extract_code_block() == "def f(): pass"

    def test_extract_code_block_no_markers(self):
        sol = Solution(id="s", agent_id="a", round_num=1,
                       code="def f(): pass")
        assert sol.extract_code_block() == "def f(): pass"

    def test_pass_rate_no_execution(self):
        sol = Solution(id="s", agent_id="a", round_num=1, code="code")
        assert sol.pass_rate == 0.0

    def test_pass_rate_with_results(self):
        sol = Solution(id="s", agent_id="a", round_num=1, code="code")
        sol.execution_result = ExecutionResult(
            status=SolutionStatus.TEST_FAILED, tests_passed=2, tests_total=4
        )
        assert sol.pass_rate == 0.5

    def test_passed_all_tests_true(self, passing_solution):
        assert passing_solution.passed_all_tests is True

    def test_passed_all_tests_false(self, failing_solution):
        assert failing_solution.passed_all_tests is False

    def test_to_dict(self, passing_solution):
        d = passing_solution.to_dict()
        assert d["id"] == passing_solution.id
        assert d["execution_result"]["tests_passed"] == 3
        assert d["is_revision"] is False


# =============================================================================
# TestAgent
# =============================================================================

class TestAgent:
    def test_properties(self, sample_agent):
        assert sample_agent.model == "qwen2.5-coder:7b"
        assert sample_agent.role == AgentRole.GENERAL
        assert sample_agent.temperature == 0.3

    def test_add_message_proposal_updates_stats(self, sample_agent):
        msg = AgentMessage(agent_id="agent_1", round_num=1,
                           message_type="proposal", content="code")
        sample_agent.add_message(msg)
        assert sample_agent.stats.solutions_proposed == 1

    def test_add_message_critique_updates_stats(self, sample_agent):
        msg = AgentMessage(
            agent_id="agent_1", round_num=2, message_type="critique",
            content="review", metadata={"bugs": ["bug1", "bug2"]},
        )
        sample_agent.add_message(msg)
        assert sample_agent.stats.critiques_given == 1
        assert sample_agent.stats.bugs_found == 2

    def test_add_message_revision_updates_stats(self, sample_agent):
        msg = AgentMessage(agent_id="agent_1", round_num=2,
                           message_type="revision", content="new code")
        sample_agent.add_message(msg)
        assert sample_agent.stats.solutions_revised == 1

    def test_get_messages_for_round(self, sample_agent):
        sample_agent.add_message(
            AgentMessage(agent_id="agent_1", round_num=1,
                         message_type="proposal", content="r1")
        )
        sample_agent.add_message(
            AgentMessage(agent_id="agent_1", round_num=2,
                         message_type="revision", content="r2")
        )
        r1_msgs = sample_agent.get_messages_for_round(1)
        assert len(r1_msgs) == 1
        assert r1_msgs[0].content == "r1"

    def test_get_latest_solution(self, sample_agent):
        sample_agent.add_message(
            AgentMessage(agent_id="agent_1", round_num=1,
                         message_type="proposal", content="first")
        )
        sample_agent.add_message(
            AgentMessage(agent_id="agent_1", round_num=2,
                         message_type="revision", content="second")
        )
        latest = sample_agent.get_latest_solution()
        assert latest.content == "second"

    def test_get_latest_solution_none(self, sample_agent):
        assert sample_agent.get_latest_solution() is None

    def test_to_dict(self, sample_agent):
        d = sample_agent.to_dict()
        assert d["id"] == "agent_1"
        assert d["model"] == "qwen2.5-coder:7b"
        assert "stats" in d


# =============================================================================
# TestAgentConfig
# =============================================================================

class TestAgentConfig:
    def test_from_dict(self):
        data = {"name": "ag", "model": "codellama:7b", "role": "judge", "temperature": 0.5}
        config = AgentConfig.from_dict(data)
        assert config.model == "codellama:7b"
        assert config.role == AgentRole.JUDGE
        assert config.temperature == 0.5

    def test_from_dict_defaults(self):
        data = {"model": "m:7b"}
        config = AgentConfig.from_dict(data)
        assert config.role == AgentRole.GENERAL
        assert config.temperature == 0.3


# =============================================================================
# TestCritique
# =============================================================================

class TestCritique:
    def test_average_rating(self, sample_critique):
        expected = (7 + 8 + 9) / 3
        assert sample_critique.average_rating == expected

    def test_critical_bugs(self):
        c = Critique(
            id="c1", agent_id="a1", solution_id="s1",
            target_agent_id="a2", round_num=2,
            bugs=[
                Bug(description="critical", severity=BugSeverity.CRITICAL),
                Bug(description="minor", severity=BugSeverity.MINOR),
            ],
        )
        assert len(c.critical_bugs) == 1
        assert c.critical_bugs[0].description == "critical"

    def test_total_issues(self):
        c = Critique(
            id="c1", agent_id="a1", solution_id="s1",
            target_agent_id="a2", round_num=2,
            bugs=[Bug(description="b1")],
            improvements=[Improvement(description="i1")],
        )
        assert c.total_issues == 2

    def test_to_dict_from_dict_roundtrip(self, sample_critique):
        d = sample_critique.to_dict()
        restored = Critique.from_dict(d)
        assert restored.id == sample_critique.id
        assert restored.correctness_rating == sample_critique.correctness_rating
        assert len(restored.bugs) == len(sample_critique.bugs)


# =============================================================================
# TestVote
# =============================================================================

class TestVote:
    def test_to_dict_from_dict_roundtrip(self):
        vote = Vote(
            id="v1", agent_id="a1", round_num=2,
            vote_type=VoteType.ADOPT, voted_solution_id="s1",
            confidence=0.9, reasoning="best one",
        )
        d = vote.to_dict()
        restored = Vote.from_dict(d)
        assert restored.vote_type == VoteType.ADOPT
        assert restored.confidence == 0.9
        assert restored.reasoning == "best one"


# =============================================================================
# TestDebate
# =============================================================================

class TestDebate:
    @pytest.fixture
    def debate(self, sample_task, agents_3):
        return Debate(id="d1", task=sample_task, agents=agents_3, max_rounds=3)

    def test_add_round(self, debate, passing_solution):
        rs = RoundSummary(round_num=1, solutions=[passing_solution],
                          critiques=[], votes=[])
        debate.add_round(rs)
        assert len(debate.rounds) == 1
        assert debate.current_round == 1

    def test_get_latest_solutions_empty(self, debate):
        assert debate.get_latest_solutions() == []

    def test_get_latest_solutions(self, debate, passing_solution):
        rs = RoundSummary(round_num=1, solutions=[passing_solution],
                          critiques=[], votes=[])
        debate.add_round(rs)
        assert debate.get_latest_solutions() == [passing_solution]

    def test_finalize_consensus(self, debate, passing_solution):
        consensus = ConsensusResult(
            reached=True, winning_solution_id=passing_solution.id,
            winning_agent_id=passing_solution.agent_id,
            consensus_ratio=0.9, round_num=2,
        )
        debate.finalize(
            status=DebateStatus.CONSENSUS_REACHED,
            final_solution=passing_solution, consensus=consensus,
        )
        assert debate.status == DebateStatus.CONSENSUS_REACHED
        assert debate.winning_agent_id == passing_solution.agent_id
        assert debate.end_time is not None

    def test_finalize_error(self, debate):
        debate.finalize(status=DebateStatus.ERROR, error_message="boom")
        assert debate.status == DebateStatus.ERROR
        assert debate.error_message == "boom"

    def test_duration_seconds(self, debate):
        assert debate.duration_seconds == 0.0
        debate.finalize(status=DebateStatus.ERROR)
        assert debate.duration_seconds > 0

    def test_total_rounds(self, debate, passing_solution):
        assert debate.total_rounds == 0
        rs = RoundSummary(round_num=1, solutions=[passing_solution],
                          critiques=[], votes=[])
        debate.add_round(rs)
        assert debate.total_rounds == 1

    def test_to_dict(self, debate):
        d = debate.to_dict()
        assert d["id"] == "d1"
        assert d["status"] == "pending"
        assert len(d["agents"]) == 3


# =============================================================================
# TestDebateConfig
# =============================================================================

class TestDebateConfig:
    def test_from_dict(self):
        data = {"max_rounds": 10, "consensus_threshold": 0.8, "execution_timeout": 60}
        config = DebateConfig.from_dict(data)
        assert config.max_rounds == 10
        assert config.consensus_threshold == 0.8
        assert config.execution_timeout == 60

    def test_from_dict_defaults(self):
        config = DebateConfig.from_dict({})
        assert config.max_rounds == 5
        assert config.consensus_threshold == 0.6
        assert config.early_stop_on_perfect is True


# =============================================================================
# TestRoundSummary
# =============================================================================

class TestRoundSummary:
    def test_compute_stats(self, passing_solution, failing_solution):
        rs = RoundSummary(
            round_num=1,
            solutions=[passing_solution, failing_solution],
            critiques=[],
            votes=[],
        )
        rs.compute_stats()
        assert rs.best_pass_rate == 1.0
        assert rs.avg_pass_rate == 0.5

    def test_compute_stats_empty(self):
        rs = RoundSummary(round_num=1, solutions=[], critiques=[], votes=[])
        rs.compute_stats()
        assert rs.best_pass_rate == 0.0
        assert rs.avg_pass_rate == 0.0


# =============================================================================
# TestBug and TestImprovement
# =============================================================================

class TestBug:
    def test_to_dict_from_dict(self):
        bug = Bug(description="off by one", severity=BugSeverity.MAJOR, line_number=42)
        d = bug.to_dict()
        restored = Bug.from_dict(d)
        assert restored.description == "off by one"
        assert restored.severity == BugSeverity.MAJOR
        assert restored.line_number == 42


class TestImprovement:
    def test_to_dict_from_dict(self):
        imp = Improvement(
            description="use list comprehension",
            improvement_type=ImprovementType.READABILITY,
            priority=3,
        )
        d = imp.to_dict()
        restored = Improvement.from_dict(d)
        assert restored.improvement_type == ImprovementType.READABILITY
        assert restored.priority == 3


# =============================================================================
# TestConsensusResult
# =============================================================================

class TestConsensusResult:
    def test_to_dict_from_dict(self):
        cr = ConsensusResult(
            reached=True, winning_solution_id="s1", winning_agent_id="a1",
            consensus_ratio=0.85, round_num=3, reason="strong agreement",
        )
        d = cr.to_dict()
        restored = ConsensusResult.from_dict(d)
        assert restored.reached is True
        assert restored.consensus_ratio == 0.85


# =============================================================================
# TestDebateMetrics
# =============================================================================

class TestDebateMetrics:
    def test_compute_derived_bug_fix_rate(self):
        m = DebateMetrics(
            debate_id="d1", task_id="t1", task_difficulty="easy",
            total_bugs_found=10, total_bugs_fixed=7,
            total_duration_seconds=100.0, total_rounds=5,
        )
        m.compute_derived_metrics()
        assert m.bug_fix_rate == 0.7
        assert m.avg_round_duration == 20.0

    def test_compute_derived_agent_rankings(self):
        stats = [
            AgentStats(agent_id="a1", model="m1", role=AgentRole.GENERAL,
                       critiques_given=5, bugs_found=3, times_won_debate=1),
            AgentStats(agent_id="a2", model="m2", role=AgentRole.GENERAL,
                       critiques_given=2, bugs_found=1),
        ]
        m = DebateMetrics(
            debate_id="d1", task_id="t1", task_difficulty="easy",
            agent_stats=stats,
        )
        m.compute_derived_metrics()
        assert m.most_active_agent == "a1"
        assert m.most_successful_agent == "a1"
        assert m.most_bugs_found_by == "a1"


# =============================================================================
# TestAgentProfile
# =============================================================================

class TestAgentProfile:
    def test_classify_aggressive_critic(self):
        p = AgentProfile(model="m", avg_bugs_found_per_debate=3.0,
                         times_changed_mind_ratio=0.1)
        p.classify_personality()
        assert p.personality_type == "aggressive_critic"

    def test_classify_passive_adopter(self):
        p = AgentProfile(model="m", times_changed_mind_ratio=0.8)
        p.classify_personality()
        assert p.personality_type == "passive_adopter"

    def test_classify_stubborn_defender(self):
        p = AgentProfile(model="m", times_defended_ratio=0.9)
        p.classify_personality()
        assert p.personality_type == "stubborn_defender"

    def test_classify_balanced(self):
        p = AgentProfile(model="m")
        p.classify_personality()
        assert p.personality_type == "balanced"
