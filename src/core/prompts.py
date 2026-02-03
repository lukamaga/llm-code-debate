"""
Prompts for LLM agents and response parsing.
"""
from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Task, Solution, Critique


# =============================================================================
# System Prompts
# =============================================================================

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
Your goal is to find bugs, inefficiencies, and suggest improvements.

When critiquing code:
1. Look for logical errors and bugs
2. Check for edge cases that might fail
3. Evaluate time and space complexity
4. Assess code readability and maintainability
5. Check PEP 8 compliance and code style
6. Look for opportunities to simplify or optimize
7. Suggest specific, actionable improvements

IMPORTANT: Even if the code passes all tests, you should still critique:
- Code style and readability
- Variable naming
- Code complexity (can it be simpler?)
- Efficiency (is there a faster/better algorithm?)
- Edge case handling

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


# =============================================================================
# Prompt Builders
# =============================================================================

def build_proposal_prompt(task: "Task") -> str:
    """Build prompt for initial solution proposal."""
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

Wrap your code in ```python and ``` markers."""


def build_critique_prompt(
    task: "Task",
    solutions: list["Solution"],
    agent_id: str,
) -> str:
    """Build prompt for critiquing solutions."""
    solutions_text = ""
    for i, sol in enumerate(solutions, 1):
        if sol.agent_id == agent_id:
            continue  # Skip own solution
        code = sol.extract_code_block()
        test_info = ""
        if sol.execution_result:
            er = sol.execution_result
            test_info = f"\nTest Results: {er.tests_passed}/{er.tests_total} passed ({sol.pass_rate:.0%})"
        solutions_text += f"""
### Solution {i} (by {sol.agent_id}){test_info}
```python
{code}
```
"""

    return f"""## Task: {task.name}

{task.description}

## Solutions to Review
{solutions_text}

## Instructions
For each solution above:
1. Identify any bugs or logical errors
2. Evaluate correctness, efficiency, and readability (1-10)
3. Suggest specific improvements

Format your response as:

### Solution 1 Analysis
**Bugs Found:**
- Bug 1: description
- Bug 2: description

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


def build_revision_prompt(
    task: "Task",
    own_solution: "Solution",
    critiques: list["Critique"],
    all_solutions: list["Solution"] | None = None,
    all_critiques: list["Critique"] | None = None,
) -> str:
    """Build prompt for revising solution based on critiques."""
    own_code = own_solution.extract_code_block()

    # Test results for own solution
    own_test_info = ""
    if own_solution.execution_result:
        er = own_solution.execution_result
        own_test_info = f"Test Results: {er.tests_passed}/{er.tests_total} passed ({own_solution.pass_rate:.0%})"

    # Critiques received
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

    # Other solutions (for potential adoption)
    other_solutions_text = ""
    if all_solutions:
        for sol in all_solutions:
            if sol.agent_id != own_solution.agent_id:
                code = sol.extract_code_block()
                test_info = ""
                if sol.execution_result:
                    er = sol.execution_result
                    test_info = f" ({er.tests_passed}/{er.tests_total} tests, {sol.pass_rate:.0%})"
                other_solutions_text += f"""
### {sol.agent_id}'s Solution{test_info}
```python
{code}
```
"""

    # Discussion summary (critiques of other solutions)
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
```python
{own_code}
```

## Critiques Received
{critiques_text if critiques_text else "No critiques received."}
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

    prompt += """
## Instructions
Based on the critiques and other solutions:

1. **Fix your solution** - Address the bugs and issues mentioned
2. **Adopt another solution** - If another solution is clearly better, you can adopt it
3. **Create a hybrid** - Combine the best parts of multiple solutions

Provide your revised (or adopted) solution wrapped in ```python and ``` markers.
"""

    return prompt


def build_voting_prompt(
    task: "Task",
    solutions: list["Solution"],
    agent_id: str,
) -> str:
    """Build prompt for voting on best solution."""
    solutions_text = ""
    for i, sol in enumerate(solutions, 1):
        code = sol.extract_code_block()
        test_info = ""
        if sol.execution_result:
            er = sol.execution_result
            test_info = f" - {er.tests_passed}/{er.tests_total} tests passed ({sol.pass_rate:.0%})"
        is_own = " (YOUR SOLUTION)" if sol.agent_id == agent_id else ""
        solutions_text += f"""
### Solution {i}{is_own}{test_info}
By: {sol.agent_id}
```python
{code}
```
"""

    return f"""## Task: {task.name}

## Final Solutions
{solutions_text}

## Voting Instructions
Choose the BEST solution based on:
1. Correctness (passes all tests)
2. Efficiency (time/space complexity)
3. Readability (clean code)

You must vote for ONE solution, even if it's your own.

Format your response EXACTLY as:
VOTE: [solution number]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""


# =============================================================================
# Response Parsers
# =============================================================================

def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.

    Handles multiple formats:
    1. ```python ... ``` blocks
    2. ``` ... ``` blocks
    3. Indented code blocks
    4. Raw Python code

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code
    """
    # Strategy 1: Look for ```python blocks
    python_pattern = r"```python\s*\n?(.*?)```"
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        # Return the longest match (most likely to be complete)
        return max(matches, key=len).strip()

    # Strategy 2: Look for generic ``` blocks
    generic_pattern = r"```\s*\n?(.*?)```"
    matches = re.findall(generic_pattern, response, re.DOTALL)
    if matches:
        # Filter for Python-looking code
        for match in matches:
            if _looks_like_python(match):
                return match.strip()
        # Return longest if no clear Python
        return max(matches, key=len).strip()

    # Strategy 3: Try to find Python code line by line
    lines = response.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        # Check if line looks like Python
        if _looks_like_python(stripped) or in_code:
            if stripped.startswith(("def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ", "    ")):
                in_code = True
                code_lines.append(line)
            elif in_code and (line.startswith("    ") or line.startswith("\t") or not stripped):
                code_lines.append(line)
            elif in_code and stripped and not stripped.startswith("#"):
                # End of code block
                break

    if code_lines:
        code = "\n".join(code_lines).strip()
        if _looks_like_python(code):
            return code

    # Strategy 4: Return entire response as fallback
    return response.strip()


def _looks_like_python(text: str) -> bool:
    """Check if text looks like valid Python code."""
    if not text.strip():
        return False

    # Quick keyword check
    python_keywords = ["def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ", "try:", "except:", "with "]
    has_keywords = any(kw in text for kw in python_keywords)

    if not has_keywords:
        return False

    # Try to parse as Python
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        # Might still be Python, just incomplete
        return has_keywords


def parse_critique_response(response: str) -> dict:
    """
    Parse critique response from LLM.

    Extracts:
    - Bugs found
    - Ratings (correctness, efficiency, readability)
    - Would adopt decision
    - Improvements

    Args:
        response: Raw critique response

    Returns:
        Parsed critique data
    """
    result = {
        "critiques": [],
        "recommendation": None,
        "adopt_solution": None,
    }

    # Split by solution sections
    solution_pattern = r"(?:###?\s*)?Solution\s*(\d+)"
    sections = re.split(solution_pattern, response, flags=re.IGNORECASE)

    # Process each solution section
    current_critique = {}
    for i, section in enumerate(sections):
        if section.strip().isdigit():
            if current_critique:
                result["critiques"].append(current_critique)
            current_critique = {"solution_num": int(section), "bugs": []}
        elif current_critique and section.strip():
            # Parse bugs
            bug_patterns = [
                r"(?:bug|error|issue|problem)[:\s]*(.+?)(?:\n|$)",
                r"[-*]\s*(.+?)(?:\n|$)",
            ]
            for pattern in bug_patterns:
                bugs = re.findall(pattern, section, re.IGNORECASE)
                for bug in bugs:
                    bug_text = bug.strip()
                    if len(bug_text) > 10 and "rating" not in bug_text.lower():
                        current_critique["bugs"].append(bug_text)

            # Parse ratings
            ratings = _parse_ratings(section)
            current_critique.update(ratings)

            # Parse would adopt
            adopt_match = re.search(r"would\s*adopt[:\s]*(yes|no)", section, re.IGNORECASE)
            if adopt_match:
                current_critique["would_adopt"] = adopt_match.group(1).lower() == "yes"

    if current_critique:
        result["critiques"].append(current_critique)

    # If no structured critiques found, create default
    if not result["critiques"]:
        result["critiques"] = [{
            "solution_num": 1,
            "bugs": [],
            "correctness_rating": 5,
            "efficiency_rating": 5,
            "readability_rating": 5,
        }]

    return result


def _parse_ratings(text: str) -> dict:
    """Extract ratings from text."""
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

    # Set defaults if not found
    for _, key in patterns:
        if key not in ratings:
            ratings[key] = 5

    return ratings


def parse_vote_response(response: str) -> dict:
    """
    Parse voting response from LLM.

    Args:
        response: Raw vote response

    Returns:
        Parsed vote data with vote_type, voted_solution, confidence, reasoning
    """
    result = {
        "vote_type": "adopt",
        "voted_solution": None,
        "confidence": 0.5,
        "reasoning": "",
    }

    # Parse VOTE line
    vote_match = re.search(r"VOTE[:\s]*(\d+)", response, re.IGNORECASE)
    if vote_match:
        result["voted_solution"] = int(vote_match.group(1))

    # Parse CONFIDENCE line
    conf_match = re.search(r"CONFIDENCE[:\s]*([\d.]+)", response, re.IGNORECASE)
    if conf_match:
        try:
            result["confidence"] = min(1.0, max(0.0, float(conf_match.group(1))))
        except ValueError:
            pass

    # Parse REASONING line
    reason_match = re.search(r"REASONING[:\s]*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()[:200]

    # Determine vote type
    if result["voted_solution"]:
        result["vote_type"] = "adopt"
    elif "abstain" in response.lower():
        result["vote_type"] = "abstain"
    elif "defend" in response.lower():
        result["vote_type"] = "defend"

    return result
