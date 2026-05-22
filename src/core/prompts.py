from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..models import Task, Solution, Critique


SYSTEM_PROMPT_CODER = """You are an expert Python programmer participating in a code review debate.
Your goal is to write correct, efficient, and readable Python code.

When proposing or revising code:
1. Write complete, working Python code
2. Follow PEP 8 style guidelines
3. Include type hints where helpful
4. Handle edge cases appropriately
5. Optimize for both correctness and efficiency

Always wrap your code in ```python and ``` markers."""


SYSTEM_PROMPT_CRITIC = """You are an expert code reviewer participating in a code review debate.
Your PRIMARY goal is to find bugs that cause test failures and explain HOW to fix them.

When critiquing code:
1. **CORRECTNESS FIRST**: If tests are failing, focus on finding WHY
2. For each bug found, explain the ROOT CAUSE (not just "there is a bug")
3. Suggest a specific FIX: what line to change and how
4. Check edge cases: empty input, boundary values, off-by-one errors
5. **THEN QUALITY**: Evaluate readability, maintainability, and code style
6. Assess time/space complexity and suggest optimizations

PRIORITY: If a solution fails tests, spend most of your analysis on correctness bugs.
If all tests pass, focus on code quality, efficiency, and readability improvements.

Be constructive and specific. Rate each solution on correctness, efficiency, and readability (1-10 scale)."""


SYSTEM_PROMPT_JUDGE = """You are an impartial judge in a code review debate.
Your goal is to evaluate solutions fairly based on multiple criteria.

When judging, consider ALL of these factors:
1. **Correctness** (40%): Does it pass all tests?
2. **Efficiency** (25%): Time/space complexity, algorithm choice
3. **Readability** (20%): Clean code, good naming, easy to understand
4. **Style** (15%): PEP 8 compliance, consistent formatting

IMPORTANT: Two solutions may both pass all tests, but one could be BETTER in:
- Simpler algorithm
- More readable variable names
- Better error handling
- Cleaner structure

Always provide clear reasoning comparing the solutions on these criteria."""


def build_proposal_prompt(task: "Task") -> str:
    return f"""## Task: {task.name}

### Description
{task.description}

### Function Signature
```python
{task.signature}
```

### Constraints
{chr(10).join(f'- {c}' for c in task.constraints)}

### Instructions
Write a complete Python implementation that solves this task.
Make sure your code handles all edge cases and follows the constraints.

CRITICAL: Submit COMPLETE working code, not a skeleton.
-  DO NOT use `pass`, `...`, `# TODO`, or `raise NotImplementedError` as a function body.
-  DO NOT outline classes/methods and leave them empty.
-  EVERY function must have real, executable logic that returns the right value.
-  Prefer a simple brute-force solution that PASSES TESTS over an elegant stub that doesn't.

Wrap your code in ```python and ``` markers."""


def build_critique_prompt(
    task: "Task",
    solutions: list["Solution"],
    agent_id: str,
    previous_critique_summary: str = "",
) -> str:
    solutions_text = ""
    counter = 0
    for sol in solutions:
        if sol.agent_id == agent_id:
            continue
        counter += 1
        test_info = ""
        test_feedback = ""
        if sol.execution_result:
            er = sol.execution_result
            if er.tests_passed == 0 and er.tests_total <= 1 and er.error_message:
                test_info = "\nERROR: Code failed to import — 0 tests could run"
            else:
                test_info = f"\nTest Results: {er.tests_passed}/{er.tests_total} passed ({sol.pass_rate:.0%})"
            test_feedback = _format_test_feedback(er)
        if sol.code_files:
            code_display = format_multi_file_code_display(sol)
        else:
            code_display = f"```python\n{sol.extract_code_block()}\n```"
        test_section = f"\n{test_feedback}" if test_feedback else ""
        solutions_text += f"""
### Solution {counter} (by {sol.agent_id}){test_info}
{code_display}
{test_section}
"""

    history_section = ""
    if previous_critique_summary:
        history_section = f"""
## Previous Round Critiques (already discussed)
{previous_critique_summary}
NOTE: Focus on issues NOT yet fixed, or identify NEW bugs. Do not repeat the same critique.
"""

    return f"""## Task: {task.name}

{task.description}

## Solutions to Review
{solutions_text}
{history_section}
## Instructions
For each solution above:
1. **Ground every bug report in concrete evidence** — cite a FAILING test name (from "Detailed test results") or a specific line of code. Do NOT invent bugs that are not supported by a failing test or a real code defect.
2. If a test is shown as FAILED with a full error, quote the key part of that error (expected vs. actual) in your bug description.
3. For each bug, explain the root cause and suggest a specific fix.
4. Evaluate correctness, efficiency, and readability (1-10).
5. If the solution already passes all shown tests, say so — do not fabricate problems to look thorough.

Format your response as:

### Solution 1 Analysis
**Bugs Found:**
- Bug 1: description — Evidence: (failing test name OR code line) — Root cause: ... — Fix: ...
- Bug 2: description — Evidence: ... — Root cause: ... — Fix: ...

**Ratings:**
- Correctness: X/10
- Efficiency: X/10
- Readability: X/10

**Improvements:**
- Improvement 1
- Improvement 2

**Would Adopt:** Yes/No
**Reason:** explanation

(Repeat for each solution)"""


REVISION_STRATEGIES: dict[str, str] = {
    "step_by_step": (
        "**Strategy: Step-by-Step Trace**\n"
        "Trace through each FAILING test case with your code:\n"
        "1. Write down the test input values\n"
        "2. Execute your code line by line mentally\n"
        "3. Track the state of all variables at each step\n"
        "4. Find the EXACT line where actual output diverges from expected\n"
        "Do NOT guess — trace precisely, then fix only the broken logic."
    ),
    "simplify": (
        "**Strategy: Simplify & Rewrite**\n"
        "Your current approach may be over-complicated or have deep structural bugs.\n"
        "1. Rewrite the solution from scratch using the SIMPLEST possible approach\n"
        "2. Use basic data structures (dict, list) before optimizing\n"
        "3. First make it CORRECT, then make it fast\n"
        "4. Start fresh — do not patch the existing code"
    ),
    "test_driven": (
        "**Strategy: Test-Driven Fix**\n"
        "Focus exclusively on the failing tests:\n"
        "1. For each failing test, identify the MINIMAL code change needed\n"
        "2. Fix one test at a time — do not refactor unrelated code\n"
        "3. After each fix, mentally re-run all tests to check for regressions\n"
        "4. Prioritize passing more tests over code elegance"
    ),
    "edge_cases": (
        "**Strategy: Edge Case Analysis**\n"
        "Focus on boundary conditions and special cases:\n"
        "1. Check: empty input, single element, maximum capacity\n"
        "2. Check: duplicate keys, zero values, negative values\n"
        "3. Check: operations in unusual order (delete before insert, etc.)\n"
        "4. For each edge case, verify your code handles it correctly"
    ),
}

STRATEGY_ORDER = ["step_by_step", "test_driven", "simplify", "edge_cases"]


def _format_test_feedback(execution_result, full_detail: bool = True) -> str:
    if not execution_result:
        return ""

    if not execution_result.test_results:
        if execution_result.error_message:
            err = execution_result.error_message
            if len(err) > 500:
                err = err[:500] + "..."
            return (
                "CRITICAL: Your code failed to import — no tests could run.\n"
                "Error:\n" + err
            )
        return ""

    lines = []
    first_failure_shown = False
    for tr in execution_result.test_results:
        if tr.passed:
            lines.append(f"  - {tr.test_name}: PASSED")
            continue

        err = tr.error_message or ""
        if full_detail and not first_failure_shown and err:
            first_failure_shown = True
            detail = err if len(err) <= 800 else err[:800] + "..."
            lines.append(f"  - {tr.test_name}: FAILED")
            lines.append("       FULL ERROR (ground truth — base your fix on this):")
            for raw_line in detail.splitlines():
                lines.append(f"      {raw_line}")
        else:
            short = err if len(err) <= 200 else err[:200] + "..."
            if short:
                lines.append(f"  - {tr.test_name}: FAILED — {short}")
            else:
                lines.append(f"  - {tr.test_name}: FAILED")

    if not lines:
        return ""

    return "Detailed test results:\n" + "\n".join(lines)


def build_revision_prompt(
    task: "Task",
    own_solution: "Solution",
    critiques: list["Critique"],
    all_solutions: list["Solution"] | None = None,
    all_critiques: list["Critique"] | None = None,
    show_all_solutions: bool = False,
    strategy: str = "",
    previous_critiques_summary: str = "",
) -> str:
    if own_solution.code_files:
        own_code_display = format_multi_file_code_display(own_solution)
    else:
        own_code_display = f"```python\n{own_solution.extract_code_block()}\n```"

    own_test_info = ""
    test_feedback = ""
    if own_solution.execution_result:
        er = own_solution.execution_result
        if er.tests_passed == 0 and er.tests_total <= 1 and er.error_message:
            own_test_info = "ERROR: Code failed to import — 0 tests could run"
        else:
            own_test_info = f"Test Results: {er.tests_passed}/{er.tests_total} passed ({own_solution.pass_rate:.0%})"
        test_feedback = _format_test_feedback(er)

    critiques_text = ""
    for crit in critiques:
        bugs = "\n".join(f"  - {b.description}" for b in crit.bugs) or "  None found"
        critiques_text += f"""
From {crit.agent_id}:
- Correctness: {crit.correctness_rating}/10
- Efficiency: {crit.efficiency_rating}/10
- Readability: {crit.readability_rating}/10
- Bugs found:
{bugs}
- Would adopt: {'Yes' if crit.would_adopt else 'No'}
"""

    other_solutions_text = ""
    if all_solutions:
        other_sols = [s for s in all_solutions if s.agent_id != own_solution.agent_id]
        if other_sols:
            if show_all_solutions:
                for sol in sorted(other_sols, key=lambda s: s.pass_rate, reverse=True):
                    test_info = ""
                    if sol.execution_result:
                        er = sol.execution_result
                        test_info = f" ({er.tests_passed}/{er.tests_total} tests, {sol.pass_rate:.0%})"
                    if sol.code_files:
                        code_display = format_multi_file_code_display(sol)
                    else:
                        code_display = f"```python\n{sol.extract_code_block()}\n```"
                    other_solutions_text += f"""
### {sol.agent_id}'s Solution{test_info}
{code_display}
"""
            else:
                best_other = max(other_sols, key=lambda s: s.pass_rate)
                test_info = ""
                if best_other.execution_result:
                    er = best_other.execution_result
                    test_info = f" ({er.tests_passed}/{er.tests_total} tests, {best_other.pass_rate:.0%})"
                if best_other.code_files:
                    code_display = format_multi_file_code_display(best_other)
                else:
                    code_display = f"```python\n{best_other.extract_code_block()}\n```"
                other_solutions_text = f"""
### {best_other.agent_id}'s Solution{test_info}
{code_display}
"""

    discussion_text = ""
    if all_critiques:
        for crit in all_critiques:
            if crit.target_agent_id != own_solution.agent_id:
                bugs_summary = ", ".join(b.description[:50] for b in crit.bugs[:3])
                if bugs_summary:
                    discussion_text += f"- {crit.agent_id} about {crit.target_agent_id}: {bugs_summary}\n"

    prompt = f"""## Task: {task.name}

{task.description}

## Your Current Solution
{own_test_info}
{own_code_display}
"""

    if test_feedback:
        prompt += f"""
## Test Feedback
{test_feedback}
"""

    if own_solution.was_truncated:
        prompt += """
## IMPORTANT: Output Was Truncated
Your previous code was CUT OFF because it exceeded the token limit.
The code is incomplete and has syntax errors. To fix this:
- Focus on the core algorithm — keep it simple and direct
- Make sure EVERY function and class is complete (no partial code)
- Prefer a straightforward approach over a complex one
"""

    prompt += f"""
## Critiques Received
{critiques_text if critiques_text else "No critiques received."}
"""

    if previous_critiques_summary:
        prompt += f"""
## Previous Round Issues (already discussed)
{previous_critiques_summary}
Focus on issues NOT yet resolved from above, or find NEW problems.
"""

    if other_solutions_text:
        prompt += f"""
## Other Solutions (you may adopt if better)
{other_solutions_text}
"""

    if discussion_text:
        prompt += f"""
## Discussion Summary
{discussion_text}
"""

    strategy_text = REVISION_STRATEGIES.get(strategy, "")
    if strategy_text:
        prompt += f"""
## Revision Strategy
{strategy_text}

"""

    prompt += """
## Instructions
Work from concrete evidence, not guesses.

1. **Start with the failing tests above** — in the "Test Feedback" section, the FIRST failing test is shown in full detail. Read its error, identify the exact expected-vs-actual mismatch, then locate the line of code that produces the wrong value. Fix THAT line.
2. **Use the critiques as hints, not commands** — only apply a critique if it is supported by a failing test OR you can verify the bug yourself in the code. Ignore suggested "bugs" that contradict passing tests.
3. **Adopt another solution** if it passes strictly more tests than yours. Do not adopt if it passes fewer.
4. **Create a hybrid** only if you can name a specific bug in your code that the other solution fixes.

## CRITICAL: Submit COMPLETE working code, not a skeleton
Your solution will be executed against the tests immediately. Stub functions FAIL.
-  DO NOT write `pass`, `...`, `# TODO`, `# Implement this`, or `raise NotImplementedError` as a function body.
-  DO NOT leave `def foo(): pass` for any function the tests will call.
-  DO NOT outline an architecture and skip the implementation.
-  EVERY function must contain real, executable logic that returns the right value.
-  If you cannot finish a complex algorithm, prefer a simple correct solution over an empty skeleton.
-  When in doubt, write a brute-force solution that passes the tests rather than an elegant stub that doesn't.

Provide your revised (or adopted) solution wrapped in ```python and ``` markers."""

    if task.is_multi_file:
        prompt += f"""

**Multi-file format:** Your solution must include ALL required files.
Use the # FILE: filename.py format for each file:

""" + "\n".join(f"# FILE: {f}" for f in task.required_files)

    prompt += "\n"

    return prompt


def build_voting_prompt(
    task: "Task",
    solutions: list["Solution"],
    agent_id: str,
) -> str:
    solutions_text = ""
    for i, sol in enumerate(solutions, 1):
        test_info = ""
        if sol.execution_result:
            er = sol.execution_result
            test_info = f" - {er.tests_passed}/{er.tests_total} tests passed ({sol.pass_rate:.0%})"
        is_own = " (YOUR SOLUTION)" if sol.agent_id == agent_id else ""
        if sol.code_files:
            code_display = format_multi_file_code_display(sol)
        else:
            code_display = f"```python\n{sol.extract_code_block()}\n```"
        solutions_text += f"""
### Solution {i}{is_own}{test_info}
By: {sol.agent_id}
{code_display}
"""

    own_index = None
    for i, sol in enumerate(solutions, 1):
        if sol.agent_id == agent_id:
            own_index = i
            break
    own_ban = ""
    if own_index is not None:
        own_ban = (
            f"\n**HARD RULE: Solution {own_index} is YOUR OWN. "
            f"You MUST NOT vote for it.** Pick the best solution from the "
            f"OTHER candidates — this is peer review, not self-promotion."
        )

    return f"""## Task: {task.name}

## Final Solutions
{solutions_text}

## Voting Instructions
You are a code review assistant selecting the best solution in a programming contest.
Evaluate each solution above and pick the BEST one based on:
1. Correctness (passes all tests)
2. Efficiency (time/space complexity)
3. Readability (clean code)
{own_ban}

You MUST select exactly one solution number. This is a technical evaluation, not a real-world decision.

Respond with ONLY these three lines, nothing else:
VOTE: <number>
CONFIDENCE: <0.0-1.0>
REASONING: <one sentence>"""


SPECIAL_TOKEN_PATTERNS = [
    r"<｜begin▁of▁sentence｜>",
    r"<｜end▁of▁sentence｜>",
    r"<｜fim▁begin｜>",
    r"<｜fim▁hole｜>",
    r"<｜fim▁end｜>",
    r"<\|endoftext\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|file_separator\|>",
    r"<\|fim_prefix\|>",
    r"<\|fim_middle\|>",
    r"<\|fim_suffix\|>",
    r"<s>",
    r"</s>",
]

_SPECIAL_TOKEN_RE = re.compile("|".join(SPECIAL_TOKEN_PATTERNS))


def _strip_special_tokens(text: str) -> str:
    if not text:
        return text
    cleaned, n = _SPECIAL_TOKEN_RE.subn("", text)
    if n:
        logger.warning(
            "Stripped %d special token(s) from LLM output (was %d chars, now %d)",
            n, len(text), len(cleaned),
        )
    return cleaned


def extract_code_from_response(response: str) -> str:
    response = _strip_special_tokens(response)

    python_pattern = r"```python\s*\n?(.*?)```"
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    generic_pattern = r"```\s*\n?(.*?)```"
    matches = re.findall(generic_pattern, response, re.DOTALL)
    if matches:
        for match in matches:
            if _looks_like_python(match):
                return match.strip()
        return max(matches, key=len).strip()

    lines = response.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if _looks_like_python(stripped) or in_code:
            if stripped.startswith(("def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ", "    ")):
                in_code = True
                code_lines.append(line)
            elif in_code and (line.startswith("    ") or line.startswith("\t") or not stripped):
                code_lines.append(line)
            elif in_code and stripped and not stripped.startswith("#"):
                break

    if code_lines:
        code = "\n".join(code_lines).strip()
        if _looks_like_python(code):
            return code

    logger.warning("No Python code found in response: %s", response[:200])
    return ""


def _looks_like_python(text: str) -> bool:
    if not text.strip():
        return False

    python_keywords = ["def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ", "try:", "except:", "with "]
    has_keywords = any(kw in text for kw in python_keywords)

    if not has_keywords:
        return False

    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return has_keywords


def parse_critique_response(response: str) -> dict:
    result = {
        "critiques": [],
        "recommendation": None,
        "adopt_solution": None,
    }

    solution_pattern = r"(?:###?\s*)?Solution\s*(\d+)"
    sections = re.split(solution_pattern, response, flags=re.IGNORECASE)

    current_critique = {}
    for i, section in enumerate(sections):
        if section.strip().isdigit():
            if current_critique:
                result["critiques"].append(current_critique)
            current_critique = {"solution_num": int(section), "bugs": []}
        elif current_critique and section.strip():
            bugs_section_match = re.search(
                r"\*?\*?bugs?\s*found\*?\*?[:\s]*(.*?)(?=\*\*(?:rating|improvement|would)|###|$)",
                section,
                re.IGNORECASE | re.DOTALL,
            )
            if bugs_section_match:
                bugs_text = bugs_section_match.group(1)
                bug_bullets = re.findall(r"[-*]\s*(.+?)(?:\n|$)", bugs_text)
                for bug in bug_bullets:
                    bug_text = bug.strip()
                    if len(bug_text) > 5 and bug_text.lower().strip() not in ("none", "none found", "no bugs", "n/a", "no bugs found"):
                        current_critique["bugs"].append(bug_text)
            else:
                keyword_bugs = re.findall(
                    r"(?:bug|error|issue|problem)[:\s]*(.+?)(?:\n|$)",
                    section,
                    re.IGNORECASE,
                )
                for bug in keyword_bugs:
                    bug_text = bug.strip()
                    if len(bug_text) > 10 and "rating" not in bug_text.lower():
                        current_critique["bugs"].append(bug_text)

            ratings, ratings_found = _parse_ratings(section)
            current_critique.update(ratings)
            current_critique["ratings_parsed"] = ratings_found

            adopt_match = re.search(r"would\s*adopt[*:\s]*(yes|no)", section, re.IGNORECASE)
            if adopt_match:
                current_critique["would_adopt"] = adopt_match.group(1).lower() == "yes"

    if current_critique:
        result["critiques"].append(current_critique)

    if not result["critiques"]:
        logger.warning("No structured critiques parsed from response: %s", response[:200])
        result["critiques"] = [{
            "solution_num": 1,
            "bugs": [],
            "correctness_rating": 5,
            "efficiency_rating": 5,
            "readability_rating": 5,
            "ratings_parsed": False,
        }]

    return result


def _parse_ratings(text: str) -> tuple[dict, bool]:
    ratings = {}

    patterns = [
        (r"correctness[:\s]*(\d+)", "correctness_rating"),
        (r"efficiency[:\s]*(\d+)", "efficiency_rating"),
        (r"readability[:\s]*(\d+)", "readability_rating"),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                ratings[key] = min(10, max(1, int(match.group(1))))
            except ValueError:
                ratings[key] = 5

    all_found = len(ratings) == len(patterns)

    for _, key in patterns:
        if key not in ratings:
            ratings[key] = 5

    return ratings, all_found


def parse_vote_response(response: str) -> dict:
    result = {
        "vote_type": "adopt",
        "voted_solution": None,
        "confidence": 0.5,
        "reasoning": "",
        "parse_failed": False,
    }

    if re.search(r"VOTE:\s*\[solution\s*number\]", response, re.IGNORECASE):
        logger.warning("Vote response is just echoing the template: %s", response[:200])
        result["parse_failed"] = True
        return result

    vote_match = re.search(
        r"\bVOTE\b\s*\*{0,2}\s*[:\-—–=>\s]*\*{0,2}\s*(\d+)",
        response, re.IGNORECASE,
    )
    if vote_match:
        result["voted_solution"] = int(vote_match.group(1))
    else:
        fallback_vote_patterns = [
            r"(?:I\s+)?vote\s+(?:for\s+)?solution\s*(\d+)",
            r"(?:choose|select|prefer)\s+solution\s*(\d+)",
            r"solution\s*(\d+)\s+is\s+(?:the\s+)?(?:best|winner|my\s+(?:choice|pick|vote))",
            r"(?:best|winning)\s+solution[:\s]*(\d+)",
            r"(?:my\s+)?(?:choice|pick)\s+(?:is\s+)?(?:solution\s*)?(\d+)",
        ]
        for pattern in fallback_vote_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["voted_solution"] = int(match.group(1))
                logger.warning(
                    "Vote parsed via fallback pattern '%s': %s",
                    pattern, response[:200],
                )
                break
        else:
            logger.warning("Failed to parse vote from response: %s", response[:200])
            result["parse_failed"] = True

    conf_match = re.search(
        r"\bCONFIDENCE\b\s*\*{0,2}\s*[:\-—–=>\s]*\*{0,2}\s*([\d.]+)",
        response, re.IGNORECASE,
    )
    if conf_match:
        try:
            result["confidence"] = min(1.0, max(0.0, float(conf_match.group(1))))
        except ValueError:
            pass
    else:
        pct_match = re.search(r"(\d+)\s*%", response)
        if pct_match:
            result["confidence"] = min(1.0, max(0.0, int(pct_match.group(1)) / 100))

    reason_match = re.search(r"REASONING[:\s]*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()[:200]

    if result["voted_solution"]:
        result["vote_type"] = "adopt"
    elif "abstain" in response.lower():
        result["vote_type"] = "abstain"
    elif "defend" in response.lower():
        result["vote_type"] = "defend"

    return result


def build_chunked_file_proposal_prompt(
    task: "Task",
    target_file: str,
    already_generated: dict[str, str] | None = None,
) -> str:
    constraints_text = chr(10).join(f'- {c}' for c in task.constraints)

    prompt = f"""## Task: {task.name}

{task.description}

### All Signatures (for reference)
```
{task.signature}
```

### Constraints
{constraints_text}
"""

    if already_generated:
        prompt += "\n### Already Implemented Files\n"
        prompt += "These files are already written. Your code MUST be compatible with them.\n"
        for fname, fcode in already_generated.items():
            prompt += f"\n# {fname}\n```python\n{fcode}\n```\n"

    prompt += f"""
### Instructions
Write ONLY the `{target_file}` module. Output a single ```python block.
- Do NOT include other files
- Make sure imports from other modules match the signatures above
- Write complete, working code — no placeholders or TODOs"""

    if already_generated:
        prompt += "\n- Your code must be compatible with the already-implemented files above"

    return prompt


def build_chunked_file_revision_prompt(
    task: "Task",
    target_file: str,
    own_solution: "Solution",
    critiques: list["Critique"],
    already_revised: dict[str, str] | None = None,
    test_feedback: str = "",
    strategy: str = "",
) -> str:
    own_code_parts = []
    if own_solution.code_files:
        for fname, fcode in own_solution.code_files.items():
            own_code_parts.append(f"# {fname}\n```python\n{fcode}\n```")
    own_code_display = "\n\n".join(own_code_parts)

    own_test_info = ""
    if own_solution.execution_result:
        er = own_solution.execution_result
        if er.tests_passed == 0 and er.tests_total <= 1 and er.error_message:
            own_test_info = "ERROR: Code failed to import — 0 tests could run"
        else:
            own_test_info = f"Test Results: {er.tests_passed}/{er.tests_total} passed ({own_solution.pass_rate:.0%})"

    critiques_text = ""
    for crit in critiques:
        bugs = "\n".join(f"  - {b.description}" for b in crit.bugs) or "  None found"
        critiques_text += f"""
From {crit.agent_id}:
- Correctness: {crit.correctness_rating}/10
- Bugs found:
{bugs}
"""

    prompt = f"""## Task: {task.name}

{task.description}

## Your Current Solution
{own_test_info}
{own_code_display}
"""

    if test_feedback:
        prompt += f"""
## Test Feedback
{test_feedback}
"""

    if own_solution.was_truncated:
        prompt += """
## IMPORTANT: Output Was Truncated
Your previous code was CUT OFF because it exceeded the token limit.
The code is incomplete and has syntax errors. To fix this:
- Focus on the core algorithm — keep it simple and direct
- Make sure EVERY function and class is complete (no partial code)
- Prefer a straightforward approach over a complex one
"""

    prompt += f"""
## Critiques Received
{critiques_text if critiques_text else "No critiques received."}
"""

    strategy_text = REVISION_STRATEGIES.get(strategy, "")
    if strategy_text:
        prompt += f"""
## Revision Strategy
{strategy_text}
"""

    if already_revised:
        prompt += "\n### Already Revised Files (this round)\n"
        prompt += "Your code MUST be compatible with these revised files.\n"
        for fname, fcode in already_revised.items():
            prompt += f"\n# {fname}\n```python\n{fcode}\n```\n"

    prompt += f"""
## Instructions
Revise ONLY `{target_file}`. Output a single ```python block.
Fix the bugs mentioned in critiques. Write complete, working code."""

    return prompt


def build_multi_file_proposal_prompt(task: "Task") -> str:
    files_list = "\n".join(f"- `{f}`" for f in task.required_files)

    constraints_text = chr(10).join(f'- {c}' for c in task.constraints)

    return f"""## Task: {task.name}

### Description
{task.description}

### Required Files
You must implement the following files:
{files_list}

### Signatures / Interfaces
```
{task.signature}
```

### Constraints
{constraints_text}

### Instructions
Write complete Python implementations for ALL required files listed above.
Each file must be a separate, complete Python module with correct imports between modules.

**IMPORTANT: Format your response with labeled code blocks like this:**

# FILE: filename1.py
```python
# your code for filename1.py
```

# FILE: filename2.py
```python
# your code for filename2.py
```

Every required file MUST be included. Make sure imports between your modules are correct."""


def extract_multi_file_code_from_response(
    response: str,
    required_files: list[str],
) -> dict[str, str]:
    response = _strip_special_tokens(response)

    code_files: dict[str, str] = {}

    file_pattern = r'#\s*FILE:\s*(\S+\.py)\s*\n\s*```(?:python)?\s*\n(.*?)```'
    matches = re.findall(file_pattern, response, re.DOTALL)
    if matches:
        for filename, code in matches:
            code_files[filename] = code.strip()

    if not all(f in code_files for f in required_files):
        alt_pattern = r'#{1,3}\s*`?(\S+\.py)`?\s*\n\s*```(?:python)?\s*\n(.*?)```'
        alt_matches = re.findall(alt_pattern, response, re.DOTALL)
        for filename, code in alt_matches:
            if filename not in code_files:
                code_files[filename] = code.strip()

    if not all(f in code_files for f in required_files):
        python_pattern = r'```python\s*\n(.*?)```'
        blocks = re.findall(python_pattern, response, re.DOTALL)
        missing = [f for f in required_files if f not in code_files]
        unassigned = [b.strip() for b in blocks if b.strip() not in code_files.values()]
        for fname, block in zip(missing, unassigned):
            code_files[fname] = block

    if not all(f in code_files for f in required_files):
        file_header_pattern = r'#\s*FILE:\s*(\S+\.py)\s*\n'
        parts = re.split(file_header_pattern, response)
        if len(parts) >= 3:
            for i in range(1, len(parts) - 1, 2):
                fname = parts[i].strip()
                code = parts[i + 1].strip()
                code = re.sub(r'^```(?:python)?\s*\n?', '', code)
                code = re.sub(r'\n?```\s*$', '', code)
                if fname in required_files and fname not in code_files and code:
                    code_files[fname] = code

    return code_files


def format_multi_file_code_display(solution: "Solution") -> str:
    if solution.code_files:
        parts = []
        files = solution.extract_code_files()
        for fname, fcode in files.items():
            parts.append(f"# FILE: {fname}\n```python\n{fcode}\n```")
        return "\n\n".join(parts)
    else:
        return f"```python\n{solution.extract_code_block()}\n```"
