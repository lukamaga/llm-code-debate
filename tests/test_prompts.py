"""
Tests for prompt building and response parsing.
"""
import pytest
from src.core.prompts import (
    extract_code_from_response,
    extract_multi_file_code_from_response,
    parse_critique_response,
    parse_vote_response,
    build_proposal_prompt,
    build_voting_prompt,
    build_critique_prompt,
    build_revision_prompt,
    _strip_special_tokens,
    _format_test_feedback,
    SPECIAL_TOKEN_PATTERNS,
)
from src.models import (
    ExecutionResult,
    Solution,
    SolutionStatus,
    Critique,
    Bug,
)
# TestResult imported under alias to avoid pytest collecting it as a test class.
from src.models import TestResult as _TestResult


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


class TestStripSpecialTokens:
    """Tests for _strip_special_tokens — tokenizer control sequences that
    models (esp. deepseek-coder) leak into generated output."""

    def test_deepseek_bos_inside_code(self):
        # Real failure pattern seen in experiments:
        # `return <｜begin▁of▁sentence｜>0` broke AST parsing.
        raw = "def answer():\n    return <｜begin▁of▁sentence｜>0"
        cleaned = _strip_special_tokens(raw)
        assert "begin" not in cleaned
        assert cleaned == "def answer():\n    return 0"

    def test_deepseek_eos(self):
        assert _strip_special_tokens("x = 1<｜end▁of▁sentence｜>") == "x = 1"

    def test_deepseek_fim_triplet(self):
        raw = "<｜fim▁begin｜>def f():<｜fim▁hole｜>    pass<｜fim▁end｜>"
        cleaned = _strip_special_tokens(raw)
        assert "fim" not in cleaned
        assert "def f():" in cleaned
        assert "pass" in cleaned

    def test_chatml_tokens(self):
        raw = "<|im_start|>assistant\nprint('hi')<|im_end|>"
        cleaned = _strip_special_tokens(raw)
        assert "im_start" not in cleaned
        assert "im_end" not in cleaned
        assert "print('hi')" in cleaned

    def test_gpt_endoftext(self):
        assert _strip_special_tokens("foo<|endoftext|>") == "foo"

    def test_bare_s_tags(self):
        raw = "<s>def g(): return 1</s>"
        cleaned = _strip_special_tokens(raw)
        assert cleaned == "def g(): return 1"

    def test_no_special_tokens_unchanged(self):
        raw = "def add(a, b):\n    return a + b  # simple"
        assert _strip_special_tokens(raw) == raw

    def test_empty_input(self):
        assert _strip_special_tokens("") == ""
        assert _strip_special_tokens(None) is None  # type: ignore[arg-type]

    def test_multiple_tokens_mixed(self):
        raw = "<｜begin▁of▁sentence｜>x = 1<|im_end|>\ny = 2<｜end▁of▁sentence｜>"
        cleaned = _strip_special_tokens(raw)
        assert cleaned == "x = 1\ny = 2"

    def test_patterns_list_exposed(self):
        # Sanity check: the public list is non-empty and contains known entries.
        assert len(SPECIAL_TOKEN_PATTERNS) >= 8
        joined = "\n".join(SPECIAL_TOKEN_PATTERNS)
        assert "begin" in joined
        assert "im_start" in joined
        assert "endoftext" in joined


class TestExtractCodeSanitization:
    """Integration: extract_code_from_response must strip special tokens
    from both the surrounding response AND from inside code blocks."""

    def test_token_inside_python_block(self):
        response = (
            "Here is the fix:\n"
            "```python\n"
            "def answer():\n"
            "    return <｜begin▁of▁sentence｜>0\n"
            "```"
        )
        code = extract_code_from_response(response)
        assert "begin" not in code
        assert "return 0" in code

    def test_token_outside_block_does_not_corrupt(self):
        response = (
            "<|im_start|>assistant\n"
            "```python\n"
            "def f(): return 1\n"
            "```<|im_end|>"
        )
        code = extract_code_from_response(response)
        assert "im_start" not in code
        assert "im_end" not in code
        assert "def f(): return 1" in code

    def test_multi_file_strips_tokens(self):
        response = (
            "# FILE: main.py\n"
            "```python\n"
            "from util import helper\n"
            "def run(): return helper()<｜end▁of▁sentence｜>\n"
            "```\n"
            "# FILE: util.py\n"
            "```python\n"
            "def helper(): return 42\n"
            "```"
        )
        files = extract_multi_file_code_from_response(response, ["main.py", "util.py"])
        assert "main.py" in files
        assert "util.py" in files
        assert "end" not in files["main.py"] or "end▁of" not in files["main.py"]
        assert "return helper()" in files["main.py"]
        assert "return 42" in files["util.py"]


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


# =============================================================================
# TIER 1 Fix #1: Test-grounded critique
# _format_test_feedback must show the FIRST failing test verbatim so critics
# and revisers work from real evidence, not hallucinated bugs.
# =============================================================================


def _make_exec_result(test_results, tests_total=None, error_message=None):
    """Helper to build an ExecutionResult from a list of (name, passed, err)."""
    trs = [
        _TestResult(test_name=n, passed=p, error_message=e)
        for (n, p, e) in test_results
    ]
    passed = sum(1 for tr in trs if tr.passed)
    total = tests_total if tests_total is not None else len(trs)
    return ExecutionResult(
        status=SolutionStatus.TEST_FAILED if passed < total else SolutionStatus.PASSED,
        tests_passed=passed,
        tests_total=total,
        test_results=trs,
        error_message=error_message,
    )


class TestTestGroundedFeedback:
    """Fix #1: _format_test_feedback(full_detail=True) must show verbatim
    error for the FIRST failing test (up to 800 chars)."""

    def test_first_failure_shown_in_full(self):
        long_err = (
            "AssertionError: assert justify(['This', 'is'], 16) == "
            "'This          is'\n"
            "  where actual = 'This is         '\n"
            "  expected    = 'This          is'\n"
            "  diff at char 5: ' ' vs ' '\n"
            "  full traceback:\n"
            "    File 'solution.py', line 42, in justify\n"
            "      result = ' '.join(words) + ' ' * padding"
        )
        er = _make_exec_result([
            ("test_two_words", False, long_err),
            ("test_three_words", False, "AssertionError: some other error"),
            ("test_edge", True, None),
        ])
        out = _format_test_feedback(er, full_detail=True)

        # First failure must be present VERBATIM (long error should survive).
        assert "where actual" in out
        assert "diff at char 5" in out
        assert "full traceback" in out
        # "Ground truth" marker must be visible so agent knows to trust it.
        assert "ground truth" in out.lower() or "FULL ERROR" in out
        # Second failure should be summarised (short form), not a new full block.
        assert "some other error" in out

    def test_very_long_error_truncated_at_800(self):
        giant = "X" * 5000
        er = _make_exec_result([("test_a", False, giant)])
        out = _format_test_feedback(er, full_detail=True)
        # Should truncate, not dump 5000 chars of X's.
        assert "..." in out
        # But at least 800 of them should survive — that's the ground-truth window.
        assert out.count("X") >= 800
        assert out.count("X") <= 900  # allow a little slack for "..." and framing

    def test_full_detail_false_preserves_legacy_200_char_limit(self):
        long_err = "AssertionError: " + ("foo bar " * 100)  # ~800 chars
        er = _make_exec_result([("test_a", False, long_err)])
        out = _format_test_feedback(er, full_detail=False)
        # In legacy mode we keep the old 200-char cap.
        assert len(out) < 400  # header + truncated line
        assert "..." in out

    def test_no_failures_no_ground_truth_block(self):
        er = _make_exec_result([
            ("test_a", True, None),
            ("test_b", True, None),
        ])
        out = _format_test_feedback(er, full_detail=True)
        assert "FULL ERROR" not in out
        assert "PASSED" in out

    def test_import_error_still_shown(self):
        er = ExecutionResult(
            status=SolutionStatus.SYNTAX_ERROR,
            tests_passed=0,
            tests_total=0,
            test_results=[],
            error_message="ImportError: No module named 'foo'",
        )
        out = _format_test_feedback(er, full_detail=True)
        assert "failed to import" in out.lower()
        assert "ImportError" in out


class TestCritiquePromptGrounding:
    """Fix #1: build_critique_prompt must instruct critics to ground bugs
    in evidence (failing test name / code line), not invent them."""

    def test_grounding_instruction_present(self, sample_task):
        sol = Solution(
            id="s1", agent_id="agent_2", round_num=1,
            code="```python\ndef foo(): return 1\n```",
        )
        sol.execution_result = _make_exec_result([
            ("test_basic", False, "AssertionError: expected 2, got 1"),
        ])
        prompt = build_critique_prompt(sample_task, [sol], agent_id="agent_1")
        low = prompt.lower()
        assert "ground" in low or "evidence" in low
        assert "failing test" in low or "failed" in low
        # Bug format should ask for Evidence: field.
        assert "evidence" in low


class TestRevisionPromptGrounding:
    """Fix #1: build_revision_prompt must tell the agent to start from the
    failing test feedback, not from the critiques alone."""

    def test_revision_instructs_start_from_failing_tests(self, sample_task):
        own = Solution(
            id="s1", agent_id="agent_1", round_num=1,
            code="```python\ndef foo(): return 1\n```",
        )
        own.execution_result = _make_exec_result([
            ("test_basic", False, "AssertionError: expected 2, got 1"),
        ])
        crit = Critique(
            id="c1", agent_id="agent_2", solution_id="s1",
            target_agent_id="agent_1", round_num=1,
            bugs=[Bug(description="off-by-one")],
            correctness_rating=5, efficiency_rating=7, readability_rating=7,
            would_adopt=False,
        )
        prompt = build_revision_prompt(sample_task, own, [crit])
        low = prompt.lower()
        # Key instruction tokens
        assert "failing test" in low
        assert ("ground" in low) or ("evidence" in low)
        # Must still carry the Test Feedback section
        assert "test feedback" in low


# =============================================================================
# TIER 1 Fix #2: Hard-disable self-vote in the prompt
# =============================================================================


class TestVotingPromptAntiSelfVote:
    """Fix #2: build_voting_prompt must include an explicit HARD RULE banning
    the voter from selecting its own solution."""

    def _two_solutions(self):
        a = Solution(id="sa", agent_id="agent_1", round_num=1,
                     code="```python\ndef a(): return 1\n```")
        a.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=3, tests_total=3,
        )
        b = Solution(id="sb", agent_id="agent_2", round_num=1,
                     code="```python\ndef b(): return 2\n```")
        b.execution_result = ExecutionResult(
            status=SolutionStatus.PASSED, tests_passed=2, tests_total=3,
        )
        return [a, b]

    def test_hard_rule_names_own_solution(self):
        sols = self._two_solutions()
        prompt = build_voting_prompt(sols[0], sols, agent_id="agent_1") if False else \
            build_voting_prompt(
                # build_voting_prompt(task, solutions, agent_id)
                task=_DummyTask(), solutions=sols, agent_id="agent_1",
            )
        assert "HARD RULE" in prompt
        assert "Solution 1" in prompt  # agent_1 is the first in solutions list
        assert "MUST NOT" in prompt
        assert "YOUR OWN" in prompt.upper()

    def test_hard_rule_uses_correct_index_when_voter_is_second(self):
        sols = self._two_solutions()
        prompt = build_voting_prompt(
            task=_DummyTask(), solutions=sols, agent_id="agent_2",
        )
        # agent_2 is second in the solutions list → Solution 2 is own.
        assert "Solution 2" in prompt
        assert "HARD RULE" in prompt

    def test_hard_rule_absent_when_voter_has_no_solution(self):
        # External judge voter (not in solutions list) — no self-vote ban needed.
        sols = self._two_solutions()
        prompt = build_voting_prompt(
            task=_DummyTask(), solutions=sols, agent_id="agent_judge_only",
        )
        assert "HARD RULE" not in prompt


class _DummyTask:
    """Minimal stand-in for Task when we only care about .name."""
    name = "dummy_task"
    description = "a task"
    difficulty = "easy"
    signature = "def f(): pass"
    tests = []
    constraints = []
    is_multi_file = False
    required_files = []
