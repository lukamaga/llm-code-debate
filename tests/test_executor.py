import pytest
from src.core.executor import CodeExecutor
from src.models import Solution, Task, SolutionStatus


@pytest.fixture
def executor():
    return CodeExecutor(timeout=10)


@pytest.fixture
def simple_task():
    return Task(
        id="add_one",
        name="Add One",
        description="Return x + 1",
        difficulty="easy",
        signature="def add_one(x: int) -> int:",
        tests=[
            "def test_basic():\n    assert add_one(1) == 2",
            "def test_zero():\n    assert add_one(0) == 1",
        ],
        constraints=[],
    )


class TestCodeExecutor:

    @pytest.mark.asyncio
    async def test_passing_solution(self, executor, simple_task):
        solution = Solution(
            id="sol_1",
            agent_id="test",
            round_num=1,
            code="```python\ndef add_one(x: int) -> int:\n    return x + 1\n```",
        )
        result = await executor.execute(solution, simple_task)
        assert result.status == SolutionStatus.PASSED
        assert result.tests_passed == 2
        assert result.all_passed

    @pytest.mark.asyncio
    async def test_failing_solution(self, executor, simple_task):
        solution = Solution(
            id="sol_2",
            agent_id="test",
            round_num=1,
            code="```python\ndef add_one(x: int) -> int:\n    return x\n```",
        )
        result = await executor.execute(solution, simple_task)
        assert result.status == SolutionStatus.TEST_FAILED
        assert result.tests_passed == 0

    @pytest.mark.asyncio
    async def test_syntax_error(self, executor, simple_task):
        solution = Solution(
            id="sol_3",
            agent_id="test",
            round_num=1,
            code="```python\ndef add_one(x:\n    return x + 1\n```",
        )
        result = await executor.execute(solution, simple_task)
        assert result.status in (SolutionStatus.SYNTAX_ERROR, SolutionStatus.RUNTIME_ERROR, SolutionStatus.TEST_FAILED)
        assert result.tests_passed == 0

    @pytest.mark.asyncio
    async def test_caching(self, executor, simple_task):
        solution = Solution(
            id="sol_4",
            agent_id="test",
            round_num=1,
            code="```python\ndef add_one(x: int) -> int:\n    return x + 1\n```",
        )
        result1 = await executor.execute(solution, simple_task)
        result2 = await executor.execute(solution, simple_task)
        assert result1.tests_passed == result2.tests_passed
