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
