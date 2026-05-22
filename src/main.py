from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.syntax import Syntax
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .core import DebateOrchestrator
from .database import DebateRepository
from .llm import MultiModelClient
from .models import AgentConfig, AgentRole, Debate, DebateConfig, RoundSummary, Task
from .analysis import MetricsCollector, DebateVisualizer

if HAS_RICH:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

logger = logging.getLogger(__name__)

if HAS_RICH:
    console = Console()
else:
    console = None


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def load_task(task_path: str) -> Task:
    with open(task_path) as f:
        data = json.load(f)
    return Task.from_dict(data)


def print_header():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold blue] LLM Code Debate System[/bold blue]\n"
            "[dim]Multi-Agent Code Generation through Debate[/dim]",
            border_style="blue",
        ))
    else:
        print("=" * 50)
        print(" LLM Code Debate System")
        print("Multi-Agent Code Generation through Debate")
        print("=" * 50)


def print_debate_result(debate: Debate, metrics: Any):
    if HAS_RICH:
        status_color = "green" if debate.status.value == "consensus_reached" else "yellow"
        console.print(Panel(
            f"[bold {status_color}]{debate.status.value.upper()}[/bold {status_color}]",
            title="Debate Status",
        ))
        
        table = Table(title=" Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Task", debate.task.name)
        table.add_row("Difficulty", debate.task.difficulty)
        table.add_row("Total Rounds", str(debate.total_rounds))
        table.add_row("Final Pass Rate", f"{metrics.final_pass_rate*100:.1f}%")
        table.add_row("Tests Passed", f"{metrics.final_tests_passed}/{metrics.final_tests_total}")
        table.add_row("Bugs Found", str(metrics.total_bugs_found))
        table.add_row("Duration", f"{debate.duration_seconds:.1f}s")
        
        if debate.winning_agent_id:
            table.add_row("Winner", debate.winning_agent_id)
        
        console.print(table)
        
        agent_table = Table(title=" Agent Statistics")
        agent_table.add_column("Agent")
        agent_table.add_column("Model")
        agent_table.add_column("Critiques")
        agent_table.add_column("Bugs Found")
        agent_table.add_column("Changed Mind")
        agent_table.add_column("Defended")
        
        for agent in debate.agents:
            is_winner = agent.id == debate.winning_agent_id
            style = "bold green" if is_winner else ""
            agent_table.add_row(
                f"{' ' if is_winner else ''}{agent.id}",
                agent.model,
                str(agent.stats.critiques_given),
                str(agent.stats.bugs_found),
                str(agent.stats.times_changed_mind),
                str(agent.stats.times_defended),
                style=style,
            )
        
        console.print(agent_table)
        
        if debate.final_solution:
            console.print("\n[bold] Final Solution:[/bold]")
            code = debate.final_solution.extract_code_block()
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(syntax)
    
    else:
        print(f"\nStatus: {debate.status.value}")
        print(f"Task: {debate.task.name} ({debate.task.difficulty})")
        print(f"Rounds: {debate.total_rounds}")
        print(f"Pass Rate: {metrics.final_pass_rate*100:.1f}%")
        print(f"Duration: {debate.duration_seconds:.1f}s")
        
        if debate.winning_agent_id:
            print(f"Winner: {debate.winning_agent_id}")
        
        print("\nAgent Statistics:")
        for agent in debate.agents:
            print(f"  {agent.id}: {agent.stats.critiques_given} critiques, "
                  f"{agent.stats.bugs_found} bugs found")
        
        if debate.final_solution:
            print("\nFinal Solution:")
            print(debate.final_solution.extract_code_block())


async def run_debate_cli(
    task_path: str,
    agents: list[str],
    max_rounds: int,
    config: dict[str, Any],
    output_dir: str | None = None,
    judge: str | None = None,
    adaptive_temperature: bool = False,
    critique_history: bool = False,
    revision_strategy: str = "uniform",
    show_all_solutions: bool = False,
) -> Debate:
    task = load_task(task_path)

    judge = (judge or "").strip() or None
    if judge and judge.lower() == "none":
        judge = None

    if HAS_RICH:
        console.print(f"\n[bold]Task:[/bold] {task.name}")
        console.print(f"[bold]Difficulty:[/bold] {task.difficulty}")
        console.print(f"[bold]Agents:[/bold] {', '.join(agents)}")
        if judge:
            console.print(f"[bold]Judge:[/bold] {judge} [dim](critiques + votes, does not propose)[/dim]")
        console.print()
    else:
        print(f"\nTask: {task.name}")
        print(f"Difficulty: {task.difficulty}")
        print(f"Agents: {', '.join(agents)}")
        if judge:
            print(f"Judge:  {judge} (critiques + votes, does not propose)")

    ollama_config = config.get("ollama", {})
    llm_client = MultiModelClient(
        base_url=ollama_config.get("base_url", "http://localhost:11434"),
        timeout=ollama_config.get("timeout", 120),
    )

    agent_configs = [
        AgentConfig(name=f"agent_{i+1}", model=model)
        for i, model in enumerate(agents)
    ]
    if judge:
        agent_configs.append(AgentConfig(
            name="judge",
            model=judge,
            role=AgentRole.JUDGE,
        ))
    
    debate_cfg = config.get("debate", {})
    debate_config = DebateConfig(
        max_rounds=max_rounds or debate_cfg.get("max_rounds", 5),
        consensus_threshold=debate_cfg.get("consensus_threshold", 0.6),
        early_stop_on_perfect=debate_cfg.get("early_stop_on_perfect", True),
        adaptive_temperature=adaptive_temperature or debate_cfg.get("adaptive_temperature", False),
        critique_history=critique_history or debate_cfg.get("critique_history", False),
        revision_strategy=revision_strategy if revision_strategy != "uniform"
                          else debate_cfg.get("revision_strategy", "uniform"),
        revision_show_all_solutions=show_all_solutions or debate_cfg.get("revision_show_all_solutions", False),
    )
    
    def on_round_complete(round_summary: RoundSummary):
        if HAS_RICH:
            console.print(
                f"[green][/green] Round {round_summary.round_num} complete: "
                f"Best pass rate: {round_summary.best_pass_rate*100:.1f}%, "
                f"Bugs found: {round_summary.bugs_found}"
            )
        else:
            print(f"Round {round_summary.round_num}: {round_summary.best_pass_rate*100:.1f}% pass rate")
    
    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        config=debate_config,
        on_round_complete=on_round_complete,
    )
    
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running debate...", total=None)
            debate = await orchestrator.run_debate(task, agent_configs)
    else:
        print("Running debate...")
        debate = await orchestrator.run_debate(task, agent_configs)
    
    await llm_client.close_all()
    
    collector = MetricsCollector()
    metrics = collector.collect_debate_metrics(debate)
    
    print_debate_result(debate, metrics)
    
    db_config = config.get("database", {})
    repository = DebateRepository(db_config.get("path", "debate_results.db"))
    repository.save_debate(debate)

    if HAS_RICH:
        console.print(f"\n[dim]Results saved to database[/dim]")

    try:
        transcript_dir = Path(output_dir) / "transcripts" if output_dir else Path("transcripts")
        visualizer = DebateVisualizer(output_dir=transcript_dir)
        transcript_path = visualizer.generate_full_transcript(debate)
        if HAS_RICH:
            console.print(f"[dim]Transcript saved to {transcript_path}[/dim]")
        else:
            print(f"Transcript saved to {transcript_path}")
    except Exception as e:
        logger.warning(f"Failed to write transcript: {e}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_path = output_path / f"{debate.id}_result.json"
        with open(json_path, "w") as f:
            json.dump(debate.to_dict(), f, indent=2, default=str)

        if HAS_RICH:
            console.print(f"[dim]Results saved to {json_path}[/dim]")

    return debate


async def run_solo_cli(
    task_path: str,
    agent_model: str,
    config: dict[str, Any],
    output_dir: str | None = None,
) -> Debate:
    task = load_task(task_path)

    if HAS_RICH:
        console.print(f"\n[bold]Task:[/bold] {task.name}")
        console.print(f"[bold]Difficulty:[/bold] {task.difficulty}")
        console.print(f"[bold]Solo agent:[/bold] {agent_model}")
        console.print()
    else:
        print(f"\nTask: {task.name}")
        print(f"Difficulty: {task.difficulty}")
        print(f"Solo agent: {agent_model}")

    ollama_config = config.get("ollama", {})
    llm_client = MultiModelClient(
        base_url=ollama_config.get("base_url", "http://localhost:11434"),
        timeout=ollama_config.get("timeout", 120),
    )

    agent_config = AgentConfig(name="agent_1", model=agent_model)

    debate_cfg = config.get("debate", {})
    debate_config = DebateConfig(
        max_rounds=1,
        consensus_threshold=1.0,
        early_stop_on_perfect=True,
    )

    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        config=debate_config,
    )

    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running solo...", total=None)
            debate = await orchestrator.run_solo(task, agent_config)
    else:
        print("Running solo...")
        debate = await orchestrator.run_solo(task, agent_config)

    await llm_client.close_all()

    collector = MetricsCollector()
    metrics = collector.collect_debate_metrics(debate)
    print_debate_result(debate, metrics)

    db_config = config.get("database", {})
    repository = DebateRepository(db_config.get("path", "debate_results.db"))
    repository.save_debate(debate)

    if HAS_RICH:
        console.print(f"\n[dim]Results saved to database[/dim]")

    try:
        transcript_dir = Path(output_dir) / "transcripts" if output_dir else Path("transcripts")
        visualizer = DebateVisualizer(output_dir=transcript_dir)
        transcript_path = visualizer.generate_full_transcript(debate)
        if HAS_RICH:
            console.print(f"[dim]Transcript saved to {transcript_path}[/dim]")
        else:
            print(f"Transcript saved to {transcript_path}")
    except Exception as e:
        logger.warning(f"Failed to write transcript: {e}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / f"{debate.id}_result.json"
        with open(json_path, "w") as f:
            json.dump(debate.to_dict(), f, indent=2, default=str)
        if HAS_RICH:
            console.print(f"[dim]Results saved to {json_path}[/dim]")

    return debate


def run_web_server(config: dict[str, Any]):
    from .web import run_server
    
    web_config = config.get("web", {})
    
    if HAS_RICH:
        console.print("\n[bold]Starting web server...[/bold]")
        console.print(f"Open [link=http://localhost:{web_config.get('port', 5000)}]http://localhost:{web_config.get('port', 5000)}[/link]")
    else:
        print(f"\nStarting web server on port {web_config.get('port', 5000)}...")
    
    run_server(
        host=web_config.get("host", "0.0.0.0"),
        port=web_config.get("port", 5000),
        debug=web_config.get("debug", False),
    )


def main():
    parser = argparse.ArgumentParser(
        description="LLM Code Debate System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --task tasks/medium/lru_cache.json
  python -m src.main --task tasks/hard/graph.json --agents qwen2.5-coder:7b deepseek-coder:6.7b
  python -m src.main --web
        """,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        help="Path to task JSON file",
    )
    
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b-instruct"],
        help="List of Ollama model names for agents",
    )

    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help=(
            "Optional heterogeneous judge model (e.g. qwen2.5-coder:32b). "
            "The judge does NOT propose or revise, but critiques all proposals "
            "and votes. Use a stronger model than the agent pool to add an "
            "external evaluator signal. Omit for no judge."
        ),
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum number of debate rounds",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web interface",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--adaptive-temperature",
        action="store_true",
        help="Increase revision temperature when pass_rate stagnates between rounds",
    )
    parser.add_argument(
        "--critique-history",
        action="store_true",
        help="Include prior-round critique summaries in the critic prompt",
    )
    parser.add_argument(
        "--revision-strategy",
        type=str,
        choices=["uniform", "diverse"],
        default="uniform",
        help=(
            "uniform = all agents get the same revision prompt (baseline). "
            "diverse = round-robin DMAD-style strategies "
            "(step_by_step, test_driven, simplify, edge_cases)."
        ),
    )
    parser.add_argument(
        "--show-all-solutions",
        action="store_true",
        help=(
            "Show all peer solutions in the revision prompt. Default is "
            "'best only' — agent sees its own + the highest-passing peer, "
            "which is cheaper and avoids copy-paste of broken peer code."
        ),
    )
    parser.add_argument(
        "--solo",
        action="store_true",
        help=(
            "Run in SOLO mode (single agent, no debate). Uses the FIRST "
            "model from --agents. Saves to the same DB/CSV pipeline as "
            "debates with mode='solo'. Used for thesis baseline comparison."
        ),
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_header()
    
    config = load_config(args.config)
    
    if args.web:
        run_web_server(config)
    
    elif args.task:
        if args.solo:
            if not args.agents:
                print("ERROR: --solo requires --agents <model>", file=sys.stderr)
                sys.exit(2)
            asyncio.run(run_solo_cli(
                task_path=args.task,
                agent_model=args.agents[0],
                config=config,
                output_dir=args.output,
            ))
        else:
            asyncio.run(run_debate_cli(
                task_path=args.task,
                agents=args.agents,
                max_rounds=args.max_rounds,
                config=config,
                output_dir=args.output,
                judge=args.judge,
                adaptive_temperature=args.adaptive_temperature,
                critique_history=args.critique_history,
                revision_strategy=args.revision_strategy,
                show_all_solutions=args.show_all_solutions,
            ))
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
