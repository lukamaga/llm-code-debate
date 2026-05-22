import pytest
from unittest.mock import AsyncMock, MagicMock

from src.models import (
    Agent,
    AgentConfig,
    AgentMessage,
    AgentRole,
    AgentStats,
    Bug,
    BugSeverity,
    CodeQualityMetrics,
    ConsensusResult,
    Critique,
    Debate,
    DebateConfig,
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
from src.llm import LLMRequest, LLMResponse, MultiModelClient
from tests.helpers import make_proposal_response, make_critique_response, make_vote_response


@pytest.fixture
def sample_task():
    return Task(
        id="test_task",
        name="Test Task",
        description="A simple test task",
        difficulty="easy",
        signature="def solution(x: int) -> int:",
        tests=["def test_basic():\n    assert solution(1) == 2"],
        constraints=["x >= 0"],
    )


@pytest.fixture
def simple_add_task():
    return Task(
        id="add_one",
        name="Add One",
        description="Write a function that adds 1 to the input",
        difficulty="easy",
        signature="def add_one(x: int) -> int:",
        tests=[
            "def test_positive():\n    assert add_one(1) == 2",
            "def test_zero():\n    assert add_one(0) == 1",
            "def test_negative():\n    assert add_one(-1) == 0",
        ],
    )


@pytest.fixture
def sample_agent_config():
    return AgentConfig(
        name="test_agent",
        model="qwen2.5-coder:7b",
    )


@pytest.fixture
def sample_agent(sample_agent_config):
    return Agent(id="agent_1", config=sample_agent_config)


@pytest.fixture
def agent_configs_3():
    return [
        AgentConfig(name="agent_1", model="qwen2.5-coder:7b"),
        AgentConfig(name="agent_2", model="deepseek-coder:6.7b"),
        AgentConfig(name="agent_3", model="codellama:7b-instruct"),
    ]


@pytest.fixture
def agents_3(agent_configs_3):
    return [
        Agent(id=f"agent_{i+1}_{c.model.split(':')[0]}", config=c)
        for i, c in enumerate(agent_configs_3)
    ]


@pytest.fixture
def judge_agent_config():
    return AgentConfig(name="judge", model="qwen2.5-coder:7b", role=AgentRole.JUDGE)


@pytest.fixture
def sample_solution(sample_task):
    return Solution(
        id="sol_1",
        agent_id="agent_1",
        round_num=1,
        code="```python\ndef solution(x: int) -> int:\n    return x + 1\n```",
    )


@pytest.fixture
def passing_solution():
    sol = Solution(
        id="sol_agent_1_r1",
        agent_id="agent_1_qwen2.5-coder",
        round_num=1,
        code="```python\ndef solution(x: int) -> int:\n    return x + 1\n```",
    )
    sol.execution_result = ExecutionResult(
        status=SolutionStatus.PASSED,
        tests_passed=3,
        tests_total=3,
    )
    return sol


@pytest.fixture
def failing_solution():
    sol = Solution(
        id="sol_agent_2_r1",
        agent_id="agent_2_deepseek-coder",
        round_num=1,
        code="```python\ndef solution(x: int) -> int:\n    return x\n```",
    )
    sol.execution_result = ExecutionResult(
        status=SolutionStatus.TEST_FAILED,
        tests_passed=0,
        tests_total=3,
    )
    return sol


@pytest.fixture
def sample_critique():
    return Critique(
        id="crit_agent_2_sol_1",
        agent_id="agent_2_deepseek-coder",
        solution_id="sol_agent_1_r1",
        target_agent_id="agent_1_qwen2.5-coder",
        round_num=2,
        bugs=[Bug(description="Missing edge case for negative input")],
        correctness_rating=7,
        efficiency_rating=8,
        readability_rating=9,
    )


@pytest.fixture
def unanimous_votes():
    return [
        Vote(
            id="v1", agent_id="agent_1_qwen2.5-coder", round_num=2,
            vote_type=VoteType.ADOPT, voted_solution_id="sol_agent_1_r1",
            voted_agent_id="agent_1_qwen2.5-coder", confidence=0.9,
        ),
        Vote(
            id="v2", agent_id="agent_2_deepseek-coder", round_num=2,
            vote_type=VoteType.ADOPT, voted_solution_id="sol_agent_1_r1",
            voted_agent_id="agent_1_qwen2.5-coder", confidence=0.85,
        ),
        Vote(
            id="v3", agent_id="agent_3_codellama", round_num=2,
            vote_type=VoteType.ADOPT, voted_solution_id="sol_agent_1_r1",
            voted_agent_id="agent_1_qwen2.5-coder", confidence=0.8,
        ),
    ]


@pytest.fixture
def debate_config():
    return DebateConfig(max_rounds=3, min_rounds=2, consensus_threshold=0.6)


@pytest.fixture
def mock_llm_client():
    client = AsyncMock(spec=MultiModelClient)

    async def _dispatch(model, request):
        prompt = request.prompt
        if "Write a complete Python implementation" in prompt:
            return make_proposal_response()
        elif "Solutions to Review" in prompt:
            return make_critique_response()
        elif "Critiques Received" in prompt or "critique" in prompt.lower():
            return make_proposal_response()
        elif "Voting Instructions" in prompt or "VOTE:" in prompt:
            return make_vote_response(solution_num=1, confidence=0.9)
        return make_proposal_response()

    client.generate = AsyncMock(side_effect=_dispatch)
    client.check_all_models = AsyncMock(return_value={"test-model": True})
    client.close_all = AsyncMock()
    return client


@pytest.fixture
def mock_execution_result():
    return ExecutionResult(
        status=SolutionStatus.PASSED,
        tests_passed=3,
        tests_total=3,
        execution_time=0.1,
    )


@pytest.fixture
def mock_quality_metrics():
    return CodeQualityMetrics(
        pylint_score=8.0,
        cyclomatic_complexity=3.0,
        maintainability_index=80.0,
    )
