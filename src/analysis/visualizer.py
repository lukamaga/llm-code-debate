"""
Visualization tools for debate analysis.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Debate, DebateMetrics


class DebateVisualizer:
    """
    Generates visualizations and reports for debates.
    """

    def __init__(self, output_dir: str | Path = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, debate: "Debate", metrics: "DebateMetrics") -> str:
        """
        Generate a text report for a debate.

        Args:
            debate: The debate object
            metrics: Collected metrics

        Returns:
            Path to generated report
        """
        report_lines = [
            f"# Debate Report: {debate.id}",
            f"Task: {debate.task.name}",
            f"Status: {debate.status.value}",
            "",
            "## Results",
            f"- Final Pass Rate: {metrics.final_pass_rate:.1%}",
            f"- Total Rounds: {metrics.total_rounds}",
            f"- Consensus Reached: {'Yes' if metrics.consensus_reached else 'No'}",
            f"- Winning Agent: {debate.winning_agent_id or 'N/A'}",
            "",
            "## Agent Performance",
        ]

        for agent in debate.agents:
            stats = agent.stats
            report_lines.extend([
                f"\n### {agent.id} ({agent.model})",
                f"- Solutions Proposed: {stats.solutions_proposed}",
                f"- Critiques Given: {stats.critiques_given}",
                f"- Bugs Found: {stats.bugs_found}",
                f"- Times Changed Mind: {stats.times_changed_mind}",
                f"- Times Defended: {stats.times_defended}",
                f"- Times Adopted Other: {stats.times_adopted_other}",
            ])

        report_lines.extend([
            "",
            "## Round Summary",
        ])

        for round_summary in debate.rounds:
            report_lines.extend([
                f"\n### Round {round_summary.round_num}",
                f"- Best Pass Rate: {round_summary.best_pass_rate:.1%}",
                f"- Average Pass Rate: {round_summary.avg_pass_rate:.1%}",
                f"- Bugs Found: {round_summary.bugs_found}",
            ])

        if debate.final_solution:
            report_lines.extend([
                "",
                "## Final Solution",
                "```python",
                debate.final_solution.extract_code_block(),
                "```",
            ])

        report = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / f"report_{debate.id}.md"
        report_path.write_text(report)

        return str(report_path)

    def export_json(self, debate: "Debate", metrics: "DebateMetrics") -> str:
        """
        Export debate data as JSON.

        Args:
            debate: The debate object
            metrics: Collected metrics

        Returns:
            Path to generated JSON file
        """
        data = {
            "debate_id": debate.id,
            "task": {
                "id": debate.task.id,
                "name": debate.task.name,
                "difficulty": debate.task.difficulty,
            },
            "status": debate.status.value,
            "metrics": {
                "final_pass_rate": metrics.final_pass_rate,
                "total_rounds": metrics.total_rounds,
                "consensus_reached": metrics.consensus_reached,
                "consensus_ratio": metrics.consensus_ratio,
                "total_critiques": metrics.total_critiques,
                "total_bugs_found": metrics.total_bugs_found,
                "improvement_over_best_initial": metrics.improvement_over_best_initial,
            },
            "agents": [
                {
                    "id": agent.id,
                    "model": agent.model,
                    "stats": {
                        "solutions_proposed": agent.stats.solutions_proposed,
                        "critiques_given": agent.stats.critiques_given,
                        "bugs_found": agent.stats.bugs_found,
                        "times_changed_mind": agent.stats.times_changed_mind,
                        "times_defended": agent.stats.times_defended,
                        "times_adopted_other": agent.stats.times_adopted_other,
                        "times_won_debate": agent.stats.times_won_debate,
                    },
                }
                for agent in debate.agents
            ],
            "rounds": [
                {
                    "round_num": r.round_num,
                    "best_pass_rate": r.best_pass_rate,
                    "avg_pass_rate": r.avg_pass_rate,
                    "bugs_found": r.bugs_found,
                }
                for r in debate.rounds
            ],
            "winning_agent": debate.winning_agent_id,
        }

        json_path = self.output_dir / f"debate_{debate.id}.json"
        json_path.write_text(json.dumps(data, indent=2))

        return str(json_path)

    def generate_full_transcript(
        self,
        debate: "Debate",
        out_dir: str | Path | None = None,
    ) -> str:
        """
        Write a human-readable .txt transcript of the full debate.

        Contains every proposal/revision code, every critique with bugs and
        ratings, every vote with reasoning, consensus per round, and the final
        solution. Pure read of existing Debate data — no side effects on the
        debate itself.

        Returns the path to the written file.
        """
        from datetime import datetime as _dt

        target_dir = Path(out_dir) if out_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        is_solo = debate.id.startswith("solo_") or len(debate.agents) == 1
        mode = "solo" if is_solo else "debate"
        difficulty = debate.task.difficulty or "unknown"
        task_id = debate.task.id or "task"
        ts = debate.start_time.strftime("%Y%m%d_%H%M%S")
        # Strip "solo_" prefix to avoid "solo_solo_xxx" in filename
        debate_id_short = debate.id[5:] if debate.id.startswith("solo_") else debate.id
        filename = f"{difficulty}_{task_id}_{mode}_{debate_id_short}_{ts}.txt"

        sep = "=" * 78
        sub = "-" * 78
        lines: list[str] = []

        # Header
        lines.append(sep)
        lines.append(f"DEBATE TRANSCRIPT  |  {debate.task.name}  ({difficulty})")
        lines.append(sep)
        lines.append(f"Debate ID:   {debate.id}")
        lines.append(f"Mode:        {mode}")
        lines.append(f"Status:      {debate.status.value}")
        lines.append(f"Started:     {debate.start_time.isoformat(timespec='seconds')}")
        if debate.end_time:
            lines.append(f"Ended:       {debate.end_time.isoformat(timespec='seconds')}")
        lines.append(f"Duration:    {debate.duration_seconds:.1f}s")
        lines.append(f"Max rounds:  {debate.max_rounds}")
        lines.append(f"Rounds run:  {debate.total_rounds}")
        agent_summary = ", ".join(f"{a.id} ({a.model})" for a in debate.agents)
        lines.append(f"Agents:      {agent_summary}")
        if debate.error_message:
            lines.append(f"Error:       {debate.error_message}")
        lines.append("")

        # Task
        lines.append(sep)
        lines.append("TASK")
        lines.append(sep)
        lines.append(f"ID: {debate.task.id}")
        lines.append(f"Difficulty: {difficulty}")
        lines.append("")
        lines.append("Description:")
        lines.append(debate.task.description)
        lines.append("")
        if debate.task.constraints:
            lines.append("Constraints:")
            for c in debate.task.constraints:
                lines.append(f"  - {c}")
            lines.append("")
        if debate.task.required_files:
            lines.append(f"Required files: {', '.join(debate.task.required_files)}")
            lines.append("")
        lines.append(f"Tests: {len(debate.task.tests)}")
        lines.append("")

        # Rounds
        for r in debate.rounds:
            is_first_round = r.round_num == 1
            phase_label = "PROPOSAL" if is_first_round else "CRITIQUE -> REVISE -> VOTE"
            lines.append(sep)
            lines.append(f"ROUND {r.round_num}  |  {phase_label}")
            lines.append(sep)
            lines.append(
                f"Best pass rate: {r.best_pass_rate:.0%}  |  "
                f"Avg: {r.avg_pass_rate:.0%}  |  "
                f"Bugs found: {r.bugs_found}  |  "
                f"Duration: {r.duration_seconds:.1f}s"
            )
            lines.append("")

            # Solutions
            if r.solutions:
                heading = "Proposals" if is_first_round else "Revisions"
                lines.append(f"### {heading}")
                lines.append("")
                for sol in r.solutions:
                    er = sol.execution_result
                    tests_info = (
                        f"{er.tests_passed}/{er.tests_total} tests "
                        f"({sol.pass_rate:.0%})"
                        if er else "no execution result"
                    )
                    trunc = " TRUNCATED" if sol.was_truncated else ""
                    revised = (
                        f"  revised_from={sol.parent_solution_id}"
                        if sol.is_revision and sol.parent_solution_id else ""
                    )
                    lines.append(sub)
                    lines.append(
                        f"[{sol.agent_id}]  {tests_info}"
                        f"  gen={sol.generation_time:.1f}s{trunc}{revised}"
                    )
                    if er and er.error_message:
                        err = er.error_message
                        if len(err) > 600:
                            err = err[:600] + "... [truncated]"
                        lines.append(f"Error: {err}")
                    if sol.quality_metrics:
                        qm = sol.quality_metrics
                        lines.append(
                            f"Quality: pylint={qm.pylint_score:.1f}/10, "
                            f"complexity={qm.cyclomatic_complexity:.1f}, "
                            f"maintainability={qm.maintainability_index:.1f}, "
                            f"LOC={qm.lines_of_code}"
                        )
                    lines.append(sub)
                    if sol.code_files:
                        for fname, fcode in sol.code_files.items():
                            lines.append(f">>> FILE: {fname}")
                            lines.append(fcode.strip() or "(empty)")
                            lines.append("")
                    else:
                        lines.append(sol.extract_code_block() or "(no code extracted)")
                    lines.append("")

            # Critiques
            if r.critiques:
                lines.append("### Critiques")
                lines.append("")
                for c in r.critiques:
                    parsed_flag = "" if c.ratings_parsed else "  [ratings fell back to defaults]"
                    lines.append(
                        f"{c.agent_id} -> {c.target_agent_id}  "
                        f"correctness={c.correctness_rating}/10  "
                        f"efficiency={c.efficiency_rating}/10  "
                        f"readability={c.readability_rating}/10  "
                        f"would_adopt={c.would_adopt}{parsed_flag}"
                    )
                    if c.bugs:
                        lines.append(f"  Bugs ({len(c.bugs)}):")
                        for b in c.bugs:
                            lines.append(f"    - [{b.severity.value}] {b.description}")
                    else:
                        lines.append("  Bugs: none")
                    if c.improvements:
                        lines.append(f"  Improvements ({len(c.improvements)}):")
                        for imp in c.improvements:
                            lines.append(
                                f"    - [{imp.improvement_type.value} p{imp.priority}] "
                                f"{imp.description}"
                            )
                    if c.overall_assessment:
                        oa = c.overall_assessment.strip()
                        if len(oa) > 800:
                            oa = oa[:800] + "... [truncated]"
                        lines.append("  Overall assessment:")
                        for ln in oa.splitlines():
                            lines.append(f"    {ln}")
                    lines.append("")

            # Votes
            if r.votes:
                lines.append("### Votes")
                lines.append("")
                for v in r.votes:
                    target = v.voted_agent_id or "-"
                    parse_flag = "  [parse failed]" if v.parse_failed else ""
                    lines.append(
                        f"{v.agent_id}  vote={v.vote_type.value}  "
                        f"target={target}  confidence={v.confidence:.2f}"
                        f"{parse_flag}"
                    )
                    if v.reasoning:
                        rs = v.reasoning.strip()
                        if len(rs) > 500:
                            rs = rs[:500] + "... [truncated]"
                        for ln in rs.splitlines():
                            lines.append(f"    {ln}")
                    lines.append("")

            # Consensus
            if r.consensus_result:
                cr = r.consensus_result
                lines.append("### Consensus")
                lines.append(
                    f"Reached: {cr.reached}  |  Ratio: {cr.consensus_ratio:.0%}  |  "
                    f"Winner: {cr.winning_agent_id or '-'}"
                )
                if cr.reason:
                    lines.append(f"Reason: {cr.reason}")
                if cr.vote_distribution:
                    lines.append(f"Vote distribution: {cr.vote_distribution}")
                lines.append("")

        # Final
        lines.append(sep)
        lines.append("FINAL RESULT")
        lines.append(sep)
        if debate.final_solution:
            fs = debate.final_solution
            final_pass = fs.pass_rate if fs.execution_result else 0.0
            lines.append(
                f"Winner: {debate.winning_agent_id or fs.agent_id}  "
                f"(round {fs.round_num}, {final_pass:.0%})"
            )
            lines.append("")
            lines.append("Final code:")
            lines.append(sub)
            if fs.code_files:
                for fname, fcode in fs.code_files.items():
                    lines.append(f">>> FILE: {fname}")
                    lines.append(fcode.strip() or "(empty)")
                    lines.append("")
            else:
                lines.append(fs.extract_code_block() or "(no code)")
        else:
            lines.append("No final solution produced.")
        lines.append("")
        lines.append(f"Transcript generated: {_dt.now().isoformat(timespec='seconds')}")

        out_path = target_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return str(out_path)

    def generate_summary_table(self, debates: list["Debate"]) -> str:
        """
        Generate a summary table for multiple debates.

        Args:
            debates: List of debates to summarize

        Returns:
            Markdown table as string
        """
        headers = ["Debate ID", "Task", "Rounds", "Pass Rate", "Winner", "Status"]
        rows = []

        for debate in debates:
            final_rate = 0.0
            if debate.final_solution and debate.final_solution.execution_result:
                final_rate = debate.final_solution.pass_rate

            rows.append([
                debate.id,
                debate.task.name[:20],
                str(len(debate.rounds)),
                f"{final_rate:.0%}",
                debate.winning_agent_id or "N/A",
                debate.status.value,
            ])

        # Build table
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
