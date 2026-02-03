"""
Tests for prompt building and response parsing.
"""
import pytest
from src.core.prompts import (
    extract_code_from_response,
    parse_critique_response,
    parse_vote_response,
    build_proposal_prompt,
)


class TestExtractCode:
    """Tests for extract_code_from_response."""

    def test_python_block(self):
        response = "Here's the solution:\n```python\ndef foo():\n    return 42\n```"
        code = extract_code_from_response(response)
        assert "def foo():" in code
        assert "return 42" in code

    def test_generic_block(self):
        response = "```\ndef bar():\n    pass\n```"
        code = extract_code_from_response(response)
        assert "def bar():" in code

    def test_no_block(self):
        response = "def baz():\n    return 1"
        code = extract_code_from_response(response)
        assert "def baz():" in code

    def test_multiple_blocks(self):
        response = "```python\ndef short():\n    pass\n```\n\n```python\ndef longer():\n    return 1\n    return 2\n```"
        code = extract_code_from_response(response)
        # Should return the longest block
        assert "longer" in code


class TestParseCritique:
    """Tests for parse_critique_response."""

    def test_basic_parse(self):
        response = """
### Solution 1 Analysis
**Bugs Found:**
- Bug: Missing edge case handling

**Ratings:**
- Correctness: 7/10
- Efficiency: 8/10
- Readability: 9/10

**Would Adopt:** No
"""
        result = parse_critique_response(response)
        assert len(result["critiques"]) >= 1
        assert result["critiques"][0].get("correctness_rating", 5) == 7

    def test_empty_response(self):
        result = parse_critique_response("")
        assert len(result["critiques"]) >= 1


class TestParseVote:
    """Tests for parse_vote_response."""

    def test_basic_vote(self):
        response = "VOTE: 2\nCONFIDENCE: 0.8\nREASONING: Best solution overall"
        result = parse_vote_response(response)
        assert result["voted_solution"] == 2
        assert result["confidence"] == 0.8
        assert "Best solution" in result["reasoning"]

    def test_missing_fields(self):
        response = "I think solution 1 is best"
        result = parse_vote_response(response)
        assert result["confidence"] == 0.5  # default


class TestBuildPrompts:
    """Tests for prompt building."""

    def test_proposal_prompt(self, sample_task):
        prompt = build_proposal_prompt(sample_task)
        assert sample_task.name in prompt
        assert sample_task.description in prompt
        assert "```python" in prompt.lower() or "signature" in prompt.lower()
