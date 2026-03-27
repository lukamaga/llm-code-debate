#!/usr/bin/env python3
"""
Run batch experiments across multiple tasks.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

from src.core import DebateOrchestrator
from src.llm import MultiModelClient
from src.models import AgentConfig, DebateConfig, Task
from src.analysis import MetricsCollector, DebateVisualizer
from src.database import DebateRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_experiment(
    task_path: Path,
    models: list[str],
    max_rounds: int,
    output_dir: Path,
    repository: DebateRepository,
) -> dict:
    """Run a single experiment."""
    # Load task
    with open(task_path) as f:
        task_data = json.load(f)
    task = Task.from_dict(task_data)

    logger.info(f"Running experiment: {task.name}")

    # Create agents
    agent_configs = [
        AgentConfig(name=f"agent_{i+1}", model=model)
        for i, model in enumerate(models)
    ]

    # Create orchestrator
    llm_client = MultiModelClient()
    config = DebateConfig(max_rounds=max_rounds)
    orchestrator = DebateOrchestrator(llm_client=llm_client, config=config)

    # Run debate
    debate = await orchestrator.run_debate(task, agent_configs)

    # Save to database
    repository.save_debate(debate)

    # Collect metrics
    collector = MetricsCollector()
    metrics = collector.collect_debate_metrics(debate)

    # Generate visualizations
    visualizer = DebateVisualizer(output_dir / "visualizations")
    visualizer.generate_report(debate, metrics)
    visualizer.export_json(debate, metrics)

    return {
        "task": task.name,
        "status": debate.status.value,
        "final_pass_rate": metrics.final_pass_rate,
        "total_rounds": metrics.total_rounds,
        "consensus_reached": metrics.consensus_reached,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run batch experiments")
    parser.add_argument("--tasks-dir", type=str, required=True, help="Directory with task JSON files")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b-instruct"])
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard", "all"], default="all")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Initialize database
    repository = DebateRepository(str(output_dir / "experiments.db"))

    # Find all tasks
    task_files = []
    if args.difficulty == "all":
        for difficulty in ["easy", "medium", "hard"]:
            diff_dir = tasks_dir / difficulty
            if diff_dir.exists():
                task_files.extend(diff_dir.glob("*.json"))
    else:
        diff_dir = tasks_dir / args.difficulty
        if diff_dir.exists():
            task_files.extend(diff_dir.glob("*.json"))

    logger.info(f"Found {len(task_files)} tasks to run")

    # Run experiments
    results = []
    for task_path in task_files:
        try:
            result = await run_experiment(
                task_path=task_path,
                models=args.models,
                max_rounds=args.max_rounds,
                output_dir=output_dir,
                repository=repository,
            )
            results.append(result)
            logger.info(f"Completed: {result['task']} - {result['final_pass_rate']:.1%}")
        except Exception as e:
            logger.error(f"Failed {task_path}: {e}")
            results.append({"task": str(task_path), "status": "error", "error": str(e)})

    # Save summary
    summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Experiments complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
