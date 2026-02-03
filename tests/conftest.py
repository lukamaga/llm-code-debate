"""
Pytest configuration and fixtures.
"""
import pytest
from src.models import Task, Solution, AgentConfig, Agent


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
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
def sample_solution(sample_task):
    """Create a sample solution for testing."""
    return Solution(
        id="sol_1",
        agent_id="agent_1",
        round_num=1,
        code="```python\ndef solution(x: int) -> int:\n    return x + 1\n```",
    )


@pytest.fixture
def sample_agent_config():
    """Create a sample agent config."""
    return AgentConfig(
        name="test_agent",
        model="qwen2.5-coder:7b",
    )


@pytest.fixture
def sample_agent(sample_agent_config):
    """Create a sample agent."""
    return Agent(id="agent_1", config=sample_agent_config)
