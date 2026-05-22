from src.llm import LLMResponse


def make_proposal_response(code="def solution(x: int) -> int:\n    return x + 1"):
    return LLMResponse(
        content=f"```python\n{code}\n```",
        model="test-model",
        tokens_used=100,
        generation_time=0.5,
    )


def make_critique_response(correctness=8, bugs=None):
    bug_text = "\n".join(f"- Bug: {b}" for b in (bugs or ["edge case missing"]))
    return LLMResponse(
        content=(
            f"### Solution 1 Analysis\n"
            f"**Bugs Found:**\n{bug_text}\n\n"
            f"**Ratings:**\n"
            f"- Correctness: {correctness}/10\n"
            f"- Efficiency: 7/10\n"
            f"- Readability: 8/10\n\n"
            f"Would adopt: no"
        ),
        model="test-model",
        tokens_used=80,
        generation_time=0.4,
    )


def make_vote_response(solution_num=1, confidence=0.9):
    return LLMResponse(
        content=(
            f"VOTE: {solution_num}\n"
            f"CONFIDENCE: {confidence}\n"
            f"REASONING: Best solution overall"
        ),
        model="test-model",
        tokens_used=30,
        generation_time=0.2,
    )
