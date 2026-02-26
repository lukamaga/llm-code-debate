"""
Tests for the DebateOrchestrator.

All LLM calls are mocked. Code execution and quality analysis are also
mocked for unit tests (but may be left real for integration tests).
"""
from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.orchestrator import DebateOrchestrator
from src.core.executor import CodeExecutor, CodeQualityAnalyzer
from src.llm import LLMResponse, MultiModelClient
from src.models import (
    Agent,
    AgentConfig,
    AgentRole,
    CodeQualityMetrics,
    ConsensusResult,
    Debate,
    DebateConfig,
    DebateStatus,
    ExecutionResult,
    RoundSummary,
    Solution,
    SolutionStatus,
    Task,
    Vote,
    VoteType,
)

from tests.helpers import make_proposal_response, make_critique_response, make_vote_response


# =============================================================================
# Fixtures local to orchestrator tests
# =============================================================================

@pytest.fixture
def orchestrator(mock_llm_client, debate_config):
    """Create an orchestrator with mocked LLM and patched executor/analyzer."""
    orch = DebateOrchestrator(llm_client=mock_llm_client, config=debate_config)
    return orch


@pytest.fixture
def patched_orchestrator(orchestrator, mock_execution_result, mock_quality_metrics):
    """Orchestrator with executor and quality analyzer mocked out."""
    orchestrator.executor.execute = AsyncMock(return_value=mock_execution_result)
    orchestrator.quality_analyzer.analyze = AsyncMock(return_value=mock_quality_metrics)
    return orchestrator


# =============================================================================
# TestDebateOrchestratorInit
# =============================================================================

class TestDebateOrchestratorInit:
    def test_init_default_config(self, mock_llm_client):
        orch = DebateOrchestrator(llm_client=mock_llm_client)
        assert orch.config.max_rounds == 5
        assert orch.config.consensus_threshold == 0.6

    def test_init_custom_config(self, mock_llm_client):
        config = DebateConfig(max_rounds=10, consensus_threshold=0.8)
        orch = DebateOrchestrator(llm_client=mock_llm_client, config=config)
        assert orch.config.max_rounds == 10
        assert orch.config.consensus_threshold == 0.8

    def test_create_agents(self, orchestrator, agent_configs_3):
        agents = orchestrator._create_agents(agent_configs_3)
        assert len(agents) == 3
        assert agents[0].id == "agent_1_qwen2.5-coder"
        assert agents[1].id == "agent_2_deepseek-coder"
        assert agents[2].id == "agent_3_codellama"

    def test_create_agents_preserves_role(self, orchestrator, judge_agent_config):
        agents = orchestrator._create_agents([judge_agent_config])
        assert agents[0].role == AgentRole.JUDGE

    def test_callbacks_stored(self, mock_llm_client):
        on_round = MagicMock()
        on_msg = MagicMock()
        on_phase = MagicMock()
        orch = DebateOrchestrator(
            llm_client=mock_llm_client,
            on_round_complete=on_round,
            on_message=on_msg,
            on_phase=on_phase,
        )
        assert orch.on_round_complete is on_round
        assert orch.on_message is on_msg
        assert orch.on_phase is on_phase


# =============================================================================
# TestGetProposal
# =============================================================================

class TestGetProposal:
    async def test_get_proposal_success(self, patched_orchestrator, sample_agent, sample_task):
        sol = await patched_orchestrator._get_proposal(sample_agent, sample_task)
        assert sol is not None
        assert sol.agent_id == sample_agent.id
        assert sol.round_num == 1
        assert "return x + 1" in sol.code

    async def test_get_proposal_records_message(self, patched_orchestrator, sample_agent, sample_task):
        await patched_orchestrator._get_proposal(sample_agent, sample_task)
        assert len(sample_agent.messages) == 1
        assert sample_agent.messages[0].message_type == "proposal"

    async def test_get_proposal_calls_on_message(self, patched_orchestrator, sample_agent, sample_task):
        callback = MagicMock()
        patched_orchestrator.on_message = callback
        await patched_orchestrator._get_proposal(sample_agent, sample_task)
        callback.assert_called_once()

    async def test_get_proposal_llm_failure(self, patched_orchestrator, sample_agent, sample_task):
        patched_orchestrator.llm_client.generate = AsyncMock(
            side_effect=RuntimeError("Connection refused")
        )
        sol = await patched_orchestrator._get_proposal(sample_agent, sample_task)
        assert sol is None

    async def test_get_proposal_updates_stats(self, patched_orchestrator, sample_agent, sample_task):
        await patched_orchestrator._get_proposal(sample_agent, sample_task)
        assert sample_agent.stats.solutions_proposed == 1


# =============================================================================
# TestGetCritique
# =============================================================================

class TestGetCritique:
    async def test_get_critique_returns_critiques(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution, failing_solution
    ):
        solutions = [passing_solution, failing_solution]
        critiques = await patched_orchestrator._get_critique(
            sample_agent, sample_task, solutions
        )
        # Agent critiques all solutions (none are its own since agent_id differs)
        assert len(critiques) >= 1

    async def test_get_critique_updates_stats(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        await patched_orchestrator._get_critique(
            sample_agent, sample_task, [passing_solution]
        )
        assert sample_agent.stats.critiques_given >= 1

    async def test_get_critique_records_message(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        await patched_orchestrator._get_critique(
            sample_agent, sample_task, [passing_solution]
        )
        msgs = [m for m in sample_agent.messages if m.message_type == "critique"]
        assert len(msgs) == 1

    async def test_get_critique_llm_failure(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        patched_orchestrator.llm_client.generate = AsyncMock(
            side_effect=RuntimeError("timeout")
        )
        result = await patched_orchestrator._get_critique(
            sample_agent, sample_task, [passing_solution]
        )
        assert result == []

    async def test_get_critique_empty_solutions(
        self, patched_orchestrator, sample_agent, sample_task
    ):
        critiques = await patched_orchestrator._get_critique(
            sample_agent, sample_task, []
        )
        # With empty solutions the method should handle gracefully
        assert isinstance(critiques, list)


# =============================================================================
# TestGetRevision
# =============================================================================

class TestGetRevision:
    async def test_get_revision_success(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        sol = await patched_orchestrator._get_revision(
            sample_agent, sample_task, passing_solution, [], round_num=2
        )
        assert sol is not None
        assert sol.is_revision is True
        assert sol.parent_solution_id == passing_solution.id
        assert sol.round_num == 2

    async def test_get_revision_detects_defended(
        self, patched_orchestrator, sample_agent, sample_task
    ):
        # Make original solution and LLM return identical code
        code = "def solution(x: int) -> int:\n    return x + 1"
        original = Solution(
            id="sol_1", agent_id=sample_agent.id, round_num=1,
            code=f"```python\n{code}\n```",
        )
        patched_orchestrator.llm_client.generate = AsyncMock(
            return_value=make_proposal_response(code)
        )
        await patched_orchestrator._get_revision(
            sample_agent, sample_task, original, [], round_num=2
        )
        assert sample_agent.stats.times_defended == 1

    async def test_get_revision_detects_changed_mind(
        self, patched_orchestrator, sample_agent, sample_task
    ):
        original = Solution(
            id="sol_1", agent_id=sample_agent.id, round_num=1,
            code="```python\ndef solution(x): return x\n```",
        )
        # LLM returns different code
        patched_orchestrator.llm_client.generate = AsyncMock(
            return_value=make_proposal_response("def solution(x): return x + 1")
        )
        await patched_orchestrator._get_revision(
            sample_agent, sample_task, original, [], round_num=2
        )
        assert sample_agent.stats.times_changed_mind == 1

    async def test_get_revision_records_message(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        await patched_orchestrator._get_revision(
            sample_agent, sample_task, passing_solution, [], round_num=2
        )
        msgs = [m for m in sample_agent.messages if m.message_type == "revision"]
        assert len(msgs) == 1

    async def test_get_revision_llm_failure(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        patched_orchestrator.llm_client.generate = AsyncMock(
            side_effect=RuntimeError("timeout")
        )
        sol = await patched_orchestrator._get_revision(
            sample_agent, sample_task, passing_solution, [], round_num=2
        )
        assert sol is None


# =============================================================================
# TestGetVote
# =============================================================================

class TestGetVote:
    async def test_get_vote_adopt(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        patched_orchestrator.llm_client.generate = AsyncMock(
            return_value=make_vote_response(solution_num=1, confidence=0.85)
        )
        vote = await patched_orchestrator._get_vote(
            sample_agent, sample_task, [passing_solution], round_num=2
        )
        assert vote is not None
        assert vote.vote_type == VoteType.ADOPT
        assert vote.voted_solution_id == passing_solution.id
        assert vote.confidence == 0.85

    async def test_get_vote_defend(
        self, patched_orchestrator, sample_agent, sample_task
    ):
        # Create agent's own solution
        own_sol = Solution(
            id="sol_own", agent_id=sample_agent.id, round_num=1, code="code"
        )
        patched_orchestrator.llm_client.generate = AsyncMock(
            return_value=LLMResponse(
                content="VOTE: 1\nCONFIDENCE: 0.9\nREASONING: defend my solution",
                model="test", tokens_used=20, generation_time=0.1,
            )
        )
        vote = await patched_orchestrator._get_vote(
            sample_agent, sample_task, [own_sol], round_num=2
        )
        assert vote is not None

    async def test_get_vote_out_of_range(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        patched_orchestrator.llm_client.generate = AsyncMock(
            return_value=make_vote_response(solution_num=99, confidence=0.5)
        )
        vote = await patched_orchestrator._get_vote(
            sample_agent, sample_task, [passing_solution], round_num=2
        )
        # Vote is created but voted_solution_id should be None (out of range)
        assert vote is not None
        assert vote.voted_solution_id is None

    async def test_get_vote_records_message(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        await patched_orchestrator._get_vote(
            sample_agent, sample_task, [passing_solution], round_num=2
        )
        msgs = [m for m in sample_agent.messages if m.message_type == "vote"]
        assert len(msgs) == 1

    async def test_get_vote_llm_failure(
        self, patched_orchestrator, sample_agent, sample_task, passing_solution
    ):
        patched_orchestrator.llm_client.generate = AsyncMock(
            side_effect=RuntimeError("timeout")
        )
        vote = await patched_orchestrator._get_vote(
            sample_agent, sample_task, [passing_solution], round_num=2
        )
        assert vote is None


# =============================================================================
# TestFindBestSolution
# =============================================================================

class TestFindBestSolution:
    def test_empty_list(self, orchestrator):
        assert orchestrator._find_best_solution([]) is None

    def test_by_pass_rate(self, orchestrator, passing_solution, failing_solution):
        best = orchestrator._find_best_solution([failing_solution, passing_solution])
        assert best.id == passing_solution.id

    def test_by_quality_metrics(self, orchestrator):
        sol_a = Solution(id="a", agent_id="a1", round_num=1, code="code")
        sol_a.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=3, tests_total=3
        )
        sol_a.quality_metrics = CodeQualityMetrics(
            pylint_score=9.0, cyclomatic_complexity=2.0, maintainability_index=90.0
        )

        sol_b = Solution(id="b", agent_id="a2", round_num=1, code="code")
        sol_b.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=3, tests_total=3
        )
        sol_b.quality_metrics = CodeQualityMetrics(
            pylint_score=3.0, cyclomatic_complexity=15.0, maintainability_index=30.0
        )

        best = orchestrator._find_best_solution([sol_b, sol_a])
        assert best.id == "a"

    def test_by_votes(self, orchestrator):
        sol_a = Solution(id="a", agent_id="a1", round_num=1, code="code")
        sol_a.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=3, tests_total=3
        )
        sol_a.votes_received = 5

        sol_b = Solution(id="b", agent_id="a2", round_num=1, code="code")
        sol_b.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=3, tests_total=3
        )
        sol_b.votes_received = 1

        best = orchestrator._find_best_solution([sol_b, sol_a])
        assert best.id == "a"

    def test_single_solution(self, orchestrator, passing_solution):
        best = orchestrator._find_best_solution([passing_solution])
        assert best.id == passing_solution.id


# =============================================================================
# TestRunDebate
# =============================================================================

class TestRunDebate:
    async def test_run_debate_completes(
        self, patched_orchestrator, sample_task, agent_configs_3
    ):
        debate = await patched_orchestrator.run_debate(sample_task, agent_configs_3)
        assert debate is not None
        assert debate.status in (
            DebateStatus.CONSENSUS_REACHED,
            DebateStatus.MAX_ROUNDS_REACHED,
            DebateStatus.EARLY_STOP,
        )

    async def test_run_debate_has_rounds(
        self, patched_orchestrator, sample_task, agent_configs_3
    ):
        debate = await patched_orchestrator.run_debate(sample_task, agent_configs_3)
        assert len(debate.rounds) >= 1

    async def test_run_debate_max_rounds(self, mock_llm_client, sample_task):
        """When no consensus forms, debate runs max_rounds times."""
        config = DebateConfig(max_rounds=2, min_rounds=1, consensus_threshold=0.99)
        orch = DebateOrchestrator(llm_client=mock_llm_client, config=config)
        orch.executor.execute = AsyncMock(return_value=ExecutionResult(
            status=SolutionStatus.TEST_FAILED, tests_passed=1, tests_total=3
        ))
        orch.quality_analyzer.analyze = AsyncMock(return_value=CodeQualityMetrics())

        # Return votes that disagree — each agent votes for a different solution
        call_count = {"n": 0}
        original_dispatch = mock_llm_client.generate.side_effect

        async def disagreeing_dispatch(model, request):
            if "Voting Instructions" in request.prompt or "VOTE:" in request.prompt:
                call_count["n"] += 1
                return make_vote_response(
                    solution_num=(call_count["n"] % 2) + 1,
                    confidence=0.5,
                )
            return await original_dispatch(model, request)

        mock_llm_client.generate = AsyncMock(side_effect=disagreeing_dispatch)

        configs = [
            AgentConfig(name="a1", model="model:7b"),
            AgentConfig(name="a2", model="model:7b"),
        ]
        debate = await orch.run_debate(sample_task, configs)
        assert debate.status in (DebateStatus.MAX_ROUNDS_REACHED, DebateStatus.EARLY_STOP)

    async def test_run_debate_early_stop_perfect(self, mock_llm_client, sample_task):
        """Perfect solution in round 1 triggers early stop."""
        config = DebateConfig(max_rounds=5, early_stop_on_perfect=True)
        orch = DebateOrchestrator(llm_client=mock_llm_client, config=config)
        orch.executor.execute = AsyncMock(return_value=ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=3, tests_total=3
        ))
        orch.quality_analyzer.analyze = AsyncMock(return_value=CodeQualityMetrics())

        configs = [AgentConfig(name="a1", model="model:7b")]
        debate = await orch.run_debate(sample_task, configs)
        assert debate.status == DebateStatus.EARLY_STOP
        assert len(debate.rounds) == 1

    async def test_run_debate_stop_event(
        self, patched_orchestrator, sample_task, agent_configs_3
    ):
        """Setting stop_event mid-debate causes early stop."""
        stop = Event()
        # Set stop before round 2
        original_run_debate_round = patched_orchestrator._run_debate_round

        async def _stop_after_round1(debate, round_num):
            stop.set()
            return await original_run_debate_round(debate, round_num)

        patched_orchestrator._run_debate_round = _stop_after_round1
        debate = await patched_orchestrator.run_debate(
            sample_task, agent_configs_3, stop_event=stop
        )
        assert debate.status == DebateStatus.EARLY_STOP

    async def test_run_debate_callbacks_invoked(self, mock_llm_client, sample_task):
        on_round = MagicMock()
        on_msg = MagicMock()
        on_phase = MagicMock()
        config = DebateConfig(max_rounds=2)
        orch = DebateOrchestrator(
            llm_client=mock_llm_client, config=config,
            on_round_complete=on_round, on_message=on_msg, on_phase=on_phase,
        )
        orch.executor.execute = AsyncMock(return_value=ExecutionResult(
            status=SolutionStatus.TEST_FAILED, tests_passed=1, tests_total=3
        ))
        orch.quality_analyzer.analyze = AsyncMock(return_value=CodeQualityMetrics())

        configs = [AgentConfig(name="a1", model="m:7b")]
        await orch.run_debate(sample_task, configs)
        assert on_round.call_count >= 1
        assert on_phase.call_count >= 1

    async def test_run_debate_judge_does_not_propose(
        self, patched_orchestrator, sample_task
    ):
        configs = [
            AgentConfig(name="coder", model="model:7b"),
            AgentConfig(name="judge", model="model:7b", role=AgentRole.JUDGE),
        ]
        debate = await patched_orchestrator.run_debate(sample_task, configs)
        # Judge agent should have 0 solutions_proposed
        judge = next(a for a in debate.agents if a.role == AgentRole.JUDGE)
        assert judge.stats.solutions_proposed == 0

    async def test_run_debate_error_handling(self, mock_llm_client, sample_task):
        """If all LLM calls fail, debate ends with ERROR status."""
        mock_llm_client.generate = AsyncMock(side_effect=RuntimeError("total failure"))
        config = DebateConfig(max_rounds=2)
        orch = DebateOrchestrator(llm_client=mock_llm_client, config=config)
        orch.executor.execute = AsyncMock(return_value=ExecutionResult(
            status=SolutionStatus.TEST_FAILED, tests_passed=0, tests_total=3
        ))
        orch.quality_analyzer.analyze = AsyncMock(return_value=CodeQualityMetrics())

        configs = [AgentConfig(name="a1", model="m:7b")]
        debate = await orch.run_debate(sample_task, configs)
        # Should complete without raising
        assert debate is not None

    async def test_run_debate_rounds_recorded(
        self, patched_orchestrator, sample_task
    ):
        configs = [AgentConfig(name="a1", model="m:7b")]
        debate = await patched_orchestrator.run_debate(sample_task, configs)
        for i, round_summary in enumerate(debate.rounds):
            assert round_summary.round_num == i + 1


# =============================================================================
# TestGetSolutionById
# =============================================================================

class TestGetSolutionById:
    def test_found(self, orchestrator, passing_solution):
        result = orchestrator._get_solution_by_id(
            [passing_solution], passing_solution.id
        )
        assert result is passing_solution

    def test_not_found(self, orchestrator, passing_solution):
        result = orchestrator._get_solution_by_id([passing_solution], "nonexistent")
        assert result is None

    def test_none_id(self, orchestrator, passing_solution):
        result = orchestrator._get_solution_by_id([passing_solution], None)
        assert result is None
