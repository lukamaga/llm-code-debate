#!/usr/bin/env python3
"""
Quick script to run a single debate.

Usage:
    python scripts/quick_run.py
    python scripts/quick_run.py --task tasks/medium/lru_cache.json
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import DebateOrchestrator
from src.llm import MultiModelClient
from src.models import AgentConfig, DebateConfig, Task, RoundSummary
from src.analysis import MetricsCollector


async def main():
    # Configuration
    import argparse
    parser = argparse.ArgumentParser(description="Quick debate run")
    parser.add_argument("--task", default="tasks/easy/fibonacci.json", help="Path to task JSON")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to use (e.g. qwen2.5-coder:7b deepseek-coder:6.7b)")
    parser.add_argument("--rounds", type=int, default=3, help="Max debate rounds")
    args = parser.parse_args()
    task_path = args.task
    
    print(f"🤖 LLM Code Debate System - Quick Run")
    print(f"=" * 50)
    
    # Load task
    print(f"\n📝 Loading task: {task_path}")
    with open(task_path) as f:
        task_data = json.load(f)
    task = Task.from_dict(task_data)
    print(f"   Task: {task.name} ({task.difficulty})")
    
    # Setup
    print(f"\n🔧 Setting up...")
    
    llm_client = MultiModelClient(
        base_url="http://localhost:11434",
        timeout=120,
    )
    
    # Check if Ollama is available
    try:
        client = llm_client.get_client("qwen2.5-coder:7b")
        available = await client.is_available()
        if not available:
            print("❌ Ollama is not running!")
            print("   Start Ollama with: ollama serve")
            return
        print("   ✓ Ollama is available")
        
        models = await client.list_models()
        print(f"   ✓ Available models: {', '.join(models[:5])}...")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return
    
    # Agent configuration
    model_list = args.models or ["qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b-instruct"]
    agent_configs = [
        AgentConfig(name=f"agent_{i+1}", model=model)
        for i, model in enumerate(model_list)
    ]
    
    print(f"\n👥 Agents:")
    for cfg in agent_configs:
        print(f"   - {cfg.name}: {cfg.model}")
    
    # Debate configuration
    config = DebateConfig(
        max_rounds=args.rounds,
        consensus_threshold=0.5,
        early_stop_on_perfect=True,
    )
    
    # Progress callback
    def on_round_complete(round_summary: RoundSummary):
        print(f"\n✓ Round {round_summary.round_num} complete:")
        print(f"   Best pass rate: {round_summary.best_pass_rate*100:.1f}%")
        print(f"   Bugs found: {round_summary.bugs_found}")
        for sol in round_summary.solutions:
            if sol.execution_result:
                print(f"   - {sol.agent_id}: {sol.execution_result.tests_passed}/{sol.execution_result.tests_total} tests")
    
    # Create orchestrator
    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        config=config,
        on_round_complete=on_round_complete,
    )
    
    # Run debate
    print(f"\n🚀 Starting debate...")
    print(f"=" * 50)
    
    debate = await orchestrator.run_debate(task, agent_configs)
    
    # Results
    print(f"\n" + "=" * 50)
    print(f"📊 RESULTS")
    print(f"=" * 50)
    print(f"   Status: {debate.status.value}")
    print(f"   Total rounds: {debate.total_rounds}")
    print(f"   Duration: {debate.duration_seconds:.1f}s")
    
    if debate.final_solution:
        print(f"\n   Final pass rate: {debate.final_solution.pass_rate*100:.1f}%")
        if debate.winning_agent_id:
            print(f"   Winner: {debate.winning_agent_id}")
        
        print(f"\n📝 Final Solution:")
        print("-" * 40)
        print(debate.final_solution.extract_code_block())
        print("-" * 40)
    
    # Agent stats
    print(f"\n👥 Agent Statistics:")
    for agent in debate.agents:
        print(f"   {agent.id}:")
        print(f"      Critiques given: {agent.stats.critiques_given}")
        print(f"      Bugs found: {agent.stats.bugs_found}")
        print(f"      Times defended: {agent.stats.times_defended}")
        print(f"      Times changed mind: {agent.stats.times_changed_mind}")
    
    # Cleanup
    await llm_client.close_all()
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
