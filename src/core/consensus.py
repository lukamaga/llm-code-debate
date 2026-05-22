from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Agent, Solution, Vote, ConsensusResult


@dataclass
class ConsensusConfig:
    threshold: float = 0.6
    min_votes: int = 2
    weight_by_confidence: bool = True
    prefer_passing: bool = True
    min_pass_rate: float = 0.5


class ConsensusDetector:

    def __init__(self, config: ConsensusConfig | None = None):
        self.config = config or ConsensusConfig()

    def detect(
        self,
        votes: list["Vote"],
        solutions: list["Solution"],
        agents: list["Agent"],
        round_num: int,
    ) -> "ConsensusResult":
        from ..models import ConsensusResult, VoteType

        if not votes:
            return ConsensusResult(
                reached=False,
                round_num=round_num,
                reason="No votes cast",
            )

        vote_counts: dict[str, float] = {}
        raw_counts: dict[str, int] = {}

        for vote in votes:
            if vote.vote_type in (VoteType.ADOPT, VoteType.DEFEND):
                sol_id = vote.voted_solution_id
                if sol_id:
                    weight = vote.confidence if self.config.weight_by_confidence else 1.0
                    vote_counts[sol_id] = vote_counts.get(sol_id, 0) + weight
                    raw_counts[sol_id] = raw_counts.get(sol_id, 0) + 1

        valid_vote_count = sum(raw_counts.values())
        if len(votes) > 0 and valid_vote_count < len(votes):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Consensus: only %d of %d votes were valid (rest were abstain/parse failures)",
                valid_vote_count, len(votes),
            )

        if not vote_counts:
            return ConsensusResult(
                reached=False,
                round_num=round_num,
                reason="No valid votes for solutions",
            )

        if self.config.prefer_passing:
            sol_map = {s.id: s for s in solutions}
            for sol_id in vote_counts:
                sol = sol_map.get(sol_id)
                if sol:
                    if sol.execution_result and sol.execution_result.all_passed:
                        vote_counts[sol_id] *= 1.5

                    if sol.quality_metrics:
                        pylint_bonus = sol.quality_metrics.pylint_score / 10.0
                        vote_counts[sol_id] *= (1 + pylint_bonus * 0.2)

        total_weight = sum(vote_counts.values())
        winner_id = max(vote_counts, key=vote_counts.get)
        winner_weight = vote_counts[winner_id]
        winner_raw = raw_counts.get(winner_id, 0)

        ratio = winner_weight / total_weight if total_weight > 0 else 0

        winner_solution = None
        winning_agent_id = None
        for sol in solutions:
            if sol.id == winner_id:
                winner_solution = sol
                winning_agent_id = sol.agent_id
                break

        winner_pass_rate = 0.0
        if winner_solution and winner_solution.execution_result:
            winner_pass_rate = winner_solution.pass_rate

        reached = (
            ratio >= self.config.threshold and
            winner_raw >= self.config.min_votes and
            winner_pass_rate >= self.config.min_pass_rate
        )

        reason = f"Consensus {'reached' if reached else 'not reached'}: {ratio:.1%} agreement"
        if not reached and winner_pass_rate < self.config.min_pass_rate:
            reason = f"No consensus: pass rate {winner_pass_rate:.0%} < {self.config.min_pass_rate:.0%} required"

        return ConsensusResult(
            reached=reached,
            winning_solution_id=winner_id if reached else None,
            winning_agent_id=winning_agent_id if reached else None,
            consensus_ratio=ratio,
            vote_distribution=raw_counts,
            round_num=round_num,
            reason=reason,
        )

    def check_perfect_solution(self, solutions: list["Solution"]) -> "Solution | None":
        for sol in solutions:
            if sol.execution_result and sol.execution_result.all_passed:
                return sol
        return None

    def should_early_stop(
        self,
        solutions: list["Solution"],
        votes: list["Vote"],
        round_num: int,
        min_rounds: int,
    ) -> tuple[bool, str]:
        from ..models import VoteType

        if round_num < min_rounds:
            return False, "Minimum rounds not reached"

        perfect = self.check_perfect_solution(solutions)
        if perfect:
            return True, f"Perfect solution found by {perfect.agent_id}"

        voted_solutions = set()
        for vote in votes:
            if vote.vote_type in (VoteType.ADOPT, VoteType.DEFEND) and vote.voted_solution_id:
                voted_solutions.add(vote.voted_solution_id)

        if len(voted_solutions) == 1 and len(votes) >= 2:
            agreed_id = next(iter(voted_solutions))
            agreed_sol = next((s for s in solutions if s.id == agreed_id), None)
            agreed_pass_rate = agreed_sol.pass_rate if agreed_sol else 0.0
            if agreed_pass_rate >= self.config.min_pass_rate:
                return True, "Unanimous agreement reached"
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Unanimous vote for solution with %.0f%% pass rate (min %.0f%%), continuing debate",
                    agreed_pass_rate * 100, self.config.min_pass_rate * 100,
                )

        return False, "No early stop condition met"
