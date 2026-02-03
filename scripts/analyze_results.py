#!/usr/bin/env python3
"""
Analyze experiment results and generate reports.

Usage:
    python scripts/analyze_results.py --input results/ --output analysis/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import DebateRepository
from src.analysis import MetricsCollector, DebateVisualizer


def load_results(results_dir: Path) -> list[dict]:
    """Load all result files from directory."""
    results = []
    for f in results_dir.glob("*_results.json"):
        with open(f) as file:
            results.append(json.load(file))
    return results


def analyze_pass_rates(results: list[dict]) -> dict:
    """Analyze pass rates across experiments."""
    analysis = {
        "by_experiment": {},
        "by_difficulty": defaultdict(list),
        "overall": [],
    }
    
    for result in results:
        exp_name = result.get("experiment_name", "Unknown")
        debates = result.get("debates", [])
        
        pass_rates = [d["pass_rate"] for d in debates if "pass_rate" in d]
        
        if pass_rates:
            analysis["by_experiment"][exp_name] = {
                "mean": sum(pass_rates) / len(pass_rates),
                "min": min(pass_rates),
                "max": max(pass_rates),
                "count": len(pass_rates),
            }
            analysis["overall"].extend(pass_rates)
            
            for d in debates:
                if "pass_rate" in d:
                    analysis["by_difficulty"][d["difficulty"]].append(d["pass_rate"])
    
    # Compute difficulty averages
    analysis["difficulty_summary"] = {
        diff: {
            "mean": sum(rates) / len(rates),
            "count": len(rates),
        }
        for diff, rates in analysis["by_difficulty"].items()
    }
    
    return analysis


def analyze_agent_behavior(results: list[dict], repository: DebateRepository) -> dict:
    """Analyze agent behavior patterns."""
    from src.analysis import MetricsCollector
    
    collector = MetricsCollector()
    
    # Get all agent stats from database
    models_seen = set()
    for result in results:
        for debate in result.get("debates", []):
            if "debate_id" in debate:
                db_debate = repository.get_debate(debate["debate_id"])
                if db_debate and db_debate.full_debate_data:
                    for agent in db_debate.full_debate_data.get("agents", []):
                        models_seen.add(agent.get("model"))
    
    # Build profiles
    profiles = {}
    for model in models_seen:
        stats = repository.get_agent_stats_by_model(model)
        if stats:
            from src.models import AgentStats
            agent_stats = [
                AgentStats(
                    agent_id=s.agent_id,
                    model=s.model,
                    role=s.role,
                    critiques_given=s.critiques_given,
                    bugs_found=s.bugs_found,
                    times_changed_mind=s.times_changed_mind,
                    times_defended=s.times_defended,
                    times_won_debate=s.times_won_debate,
                )
                for s in stats
            ]
            profile = collector.build_agent_profile(model, agent_stats)
            profiles[model] = profile.to_dict()
    
    return profiles


def compare_with_baseline(
    experiment_results: list[dict],
    baseline_path: Path,
) -> dict:
    """Compare multi-agent results with single-agent baseline."""
    if not baseline_path.exists():
        return {"error": "Baseline results not found"}
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    baseline_by_task = {b["task_id"]: b for b in baseline if "pass_rate" in b}
    
    comparison = {
        "improvements": [],
        "summary": {},
    }
    
    for result in experiment_results:
        for debate in result.get("debates", []):
            task_id = debate.get("task_id")
            if task_id in baseline_by_task and "pass_rate" in debate:
                baseline_rate = baseline_by_task[task_id]["pass_rate"]
                multi_rate = debate["pass_rate"]
                
                improvement = multi_rate - baseline_rate
                comparison["improvements"].append({
                    "task_id": task_id,
                    "task_name": debate.get("task_name"),
                    "baseline": baseline_rate,
                    "multi_agent": multi_rate,
                    "improvement": improvement,
                    "improvement_pct": (improvement / baseline_rate * 100) if baseline_rate > 0 else float('inf'),
                })
    
    if comparison["improvements"]:
        improvements = [i["improvement"] for i in comparison["improvements"]]
        comparison["summary"] = {
            "avg_improvement": sum(improvements) / len(improvements),
            "tasks_improved": sum(1 for i in improvements if i > 0),
            "tasks_same": sum(1 for i in improvements if i == 0),
            "tasks_worse": sum(1 for i in improvements if i < 0),
            "total_tasks": len(improvements),
        }
    
    return comparison


def generate_report(
    results: list[dict],
    analysis: dict,
    agent_profiles: dict,
    comparison: dict,
    output_dir: Path,
):
    """Generate markdown report."""
    report = []
    
    report.append("# LLM Code Debate - Experiment Analysis Report\n")
    report.append(f"Generated from {len(results)} experiment(s)\n")
    
    # Overall Results
    report.append("## Overall Results\n")
    
    if analysis["overall"]:
        avg_pass = sum(analysis["overall"]) / len(analysis["overall"])
        report.append(f"- **Average Pass Rate**: {avg_pass*100:.1f}%")
        report.append(f"- **Total Debates**: {len(analysis['overall'])}")
    
    # By Experiment
    report.append("\n## Results by Experiment\n")
    report.append("| Experiment | Avg Pass Rate | Min | Max | Count |")
    report.append("|------------|---------------|-----|-----|-------|")
    
    for exp_name, stats in analysis["by_experiment"].items():
        report.append(
            f"| {exp_name} | {stats['mean']*100:.1f}% | "
            f"{stats['min']*100:.1f}% | {stats['max']*100:.1f}% | "
            f"{stats['count']} |"
        )
    
    # By Difficulty
    report.append("\n## Results by Difficulty\n")
    report.append("| Difficulty | Avg Pass Rate | Count |")
    report.append("|------------|---------------|-------|")
    
    for diff, stats in analysis.get("difficulty_summary", {}).items():
        report.append(f"| {diff} | {stats['mean']*100:.1f}% | {stats['count']} |")
    
    # Agent Profiles
    if agent_profiles:
        report.append("\n## Agent Behavior Profiles\n")
        report.append("| Model | Win Rate | Critiques/Debate | Bugs Found/Debate | Personality |")
        report.append("|-------|----------|------------------|-------------------|-------------|")
        
        for model, profile in agent_profiles.items():
            report.append(
                f"| {model} | {profile.get('win_rate', 0)*100:.1f}% | "
                f"{profile.get('avg_critiques_per_debate', 0):.1f} | "
                f"{profile.get('avg_bugs_found_per_debate', 0):.1f} | "
                f"{profile.get('personality_type', 'unknown')} |"
            )
    
    # Comparison with Baseline
    if comparison and "summary" in comparison:
        report.append("\n## Comparison with Single-Agent Baseline\n")
        summary = comparison["summary"]
        report.append(f"- **Average Improvement**: {summary.get('avg_improvement', 0)*100:.1f}%")
        report.append(f"- **Tasks Improved**: {summary.get('tasks_improved', 0)}/{summary.get('total_tasks', 0)}")
        report.append(f"- **Tasks Same**: {summary.get('tasks_same', 0)}")
        report.append(f"- **Tasks Worse**: {summary.get('tasks_worse', 0)}")
    
    # Key Findings
    report.append("\n## Key Findings\n")
    
    # Auto-generate some findings
    if analysis["by_experiment"]:
        best_exp = max(analysis["by_experiment"].items(), key=lambda x: x[1]["mean"])
        report.append(f"1. **Best Configuration**: {best_exp[0]} achieved {best_exp[1]['mean']*100:.1f}% average pass rate")
    
    if analysis.get("difficulty_summary"):
        by_diff = analysis["difficulty_summary"]
        if "easy" in by_diff and "hard" in by_diff:
            diff_gap = by_diff["easy"]["mean"] - by_diff["hard"]["mean"]
            report.append(f"2. **Difficulty Gap**: Easy tasks outperform hard by {diff_gap*100:.1f}%")
    
    if agent_profiles:
        most_bugs = max(agent_profiles.items(), key=lambda x: x[1].get("avg_bugs_found_per_debate", 0))
        report.append(f"3. **Best Bug Finder**: {most_bugs[0]} finds {most_bugs[1]['avg_bugs_found_per_debate']:.1f} bugs/debate on average")
    
    # Write report
    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(report))
    
    print(f"Report saved to {report_path}")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results"),
        help="Input directory with results",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis"),
        help="Output directory for analysis",
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="debate_results.db",
        help="Database path",
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    
    args = parser.parse_args()
    
    # Setup
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(args.input)
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Initialize repository
    repository = DebateRepository(args.db_path)
    
    # Run analyses
    print("Analyzing pass rates...")
    pass_analysis = analyze_pass_rates(results)
    
    print("Analyzing agent behavior...")
    agent_profiles = analyze_agent_behavior(results, repository)
    
    print("Comparing with baseline...")
    baseline_path = args.input / "baseline_results.json"
    comparison = compare_with_baseline(results, baseline_path)
    
    # Save raw analysis
    analysis_file = args.output / "analysis_data.json"
    with open(analysis_file, "w") as f:
        json.dump({
            "pass_analysis": pass_analysis,
            "agent_profiles": agent_profiles,
            "comparison": comparison,
        }, f, indent=2, default=str)
    
    # Generate report
    print("Generating report...")
    generate_report(
        results=results,
        analysis=pass_analysis,
        agent_profiles=agent_profiles,
        comparison=comparison,
        output_dir=args.output,
    )
    
    # Generate visualizations
    if args.visualize:
        print("Generating visualizations...")
        visualizer = DebateVisualizer(output_dir=args.output / "charts")
        
        # Load debates from database for visualization
        for result in results:
            for debate_info in result.get("debates", []):
                if "debate_id" in debate_info:
                    db_record = repository.get_debate(debate_info["debate_id"])
                    if db_record and db_record.full_debate_data:
                        # Reconstruct debate for visualization
                        # This is simplified - full implementation would use the data
                        pass
    
    print(f"\nAnalysis complete! Check {args.output}/")


if __name__ == "__main__":
    main()
