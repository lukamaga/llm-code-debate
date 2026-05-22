from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.orchestrator import DebateOrchestrator
from src.database.repository import DebateRepository
from src.llm import LLMResponse
from src.models import (
    AgentConfig,
    CodeQualityMetrics,
    DebateConfig,
    DebateStatus,
    Task,
)
from tests.helpers import make_critique_response, make_vote_response


def _make_smart_llm_mock(good_code, bad_code=None):
    from src.llm.ollama_client import MultiModelClient

    client = AsyncMock(spec=MultiModelClient)
    proposal_count = {"n": 0}

    async def dispatch(model, request):
        prompt = request.prompt
        if "Write a complete Python implementation" in prompt:
            proposal_count["n"] += 1
            code = good_code if (proposal_count["n"] % 2 == 1 or bad_code is None) else bad_code
            return LLMResponse(
                content=f"```python\n{code}\n```",
                model=model, tokens_used=100, generation_time=0.3,
            )
        elif "Solutions to Review" in prompt:
            return make_critique_response(correctness=8)
        elif "Critiques Received" in prompt or "critique" in prompt.lower():
            return LLMResponse(
                content=f"```python\n{good_code}\n```",
                model=model, tokens_used=80, generation_time=0.3,
            )
        elif "Voting Instructions" in prompt or "VOTE:" in prompt:
            return make_vote_response(solution_num=1, confidence=0.9)
        return LLMResponse(
            content=f"```python\n{good_code}\n```",
            model=model, tokens_used=50, generation_time=0.2,
        )

    client.generate = AsyncMock(side_effect=dispatch)
    client.close_all = AsyncMock()
    return client


class TestIntegration:
    @pytest.fixture
    def add_one_task(self):
        return Task(
            id="add_one", name="Add One", difficulty="easy",
            description="Write add_one(x) that returns x + 1",
            signature="def add_one(x: int) -> int:",
            tests=[
                "def test_positive():\n    assert add_one(1) == 2",
                "def test_zero():\n    assert add_one(0) == 1",
                "def test_negative():\n    assert add_one(-1) == 0",
            ],
        )

    async def test_full_debate_with_real_executor(self, add_one_task):
        good_code = "def add_one(x: int) -> int:\n    return x + 1"
        mock_llm = _make_smart_llm_mock(good_code)

        config = DebateConfig(max_rounds=3, early_stop_on_perfect=True)
        orch = DebateOrchestrator(llm_client=mock_llm, config=config)

        configs = [
            AgentConfig(name="a1", model="model-a:7b"),
            AgentConfig(name="a2", model="model-b:7b"),
        ]
        debate = await orch.run_debate(add_one_task, configs)

        assert debate.status in (DebateStatus.EARLY_STOP, DebateStatus.CONSENSUS_REACHED)
        assert debate.final_solution is not None
        assert debate.final_solution.pass_rate == 1.0
        assert len(debate.rounds) >= 1

    async def test_full_debate_saves_to_db(self, add_one_task):
        good_code = "def add_one(x: int) -> int:\n    return x + 1"
        mock_llm = _make_smart_llm_mock(good_code)

        config = DebateConfig(max_rounds=2, early_stop_on_perfect=True)
        orch = DebateOrchestrator(llm_client=mock_llm, config=config)

        configs = [AgentConfig(name="a1", model="m:7b")]
        debate = await orch.run_debate(add_one_task, configs)

        repo = DebateRepository(db_path=":memory:")
        repo.save_debate(debate)

        record = repo.get_debate(debate.id)
        assert record is not None

    async def test_improvement_over_rounds(self, add_one_task):
        good_code = "def add_one(x: int) -> int:\n    return x + 1"
        bad_code = "def add_one(x: int) -> int:\n    return x"

        mock_llm = _make_smart_llm_mock(good_code, bad_code=bad_code)

        config = DebateConfig(max_rounds=3, early_stop_on_perfect=False)
        orch = DebateOrchestrator(llm_client=mock_llm, config=config)

        configs = [
            AgentConfig(name="a1", model="m1:7b"),
            AgentConfig(name="a2", model="m2:7b"),
        ]
        debate = await orch.run_debate(add_one_task, configs)

        assert debate is not None
        assert len(debate.rounds) >= 1
        all_sols = debate.all_solutions
        any_passing = any(s.pass_rate == 1.0 for s in all_sols)
        assert any_passing

    async def test_single_agent_debate(self, add_one_task):
        good_code = "def add_one(x: int) -> int:\n    return x + 1"
        mock_llm = _make_smart_llm_mock(good_code)

        config = DebateConfig(max_rounds=2, early_stop_on_perfect=True)
        orch = DebateOrchestrator(llm_client=mock_llm, config=config)

        configs = [AgentConfig(name="solo", model="m:7b")]
        debate = await orch.run_debate(add_one_task, configs)

        assert debate is not None
        assert debate.status in (
            DebateStatus.EARLY_STOP,
            DebateStatus.CONSENSUS_REACHED,
            DebateStatus.MAX_ROUNDS_REACHED,
        )
