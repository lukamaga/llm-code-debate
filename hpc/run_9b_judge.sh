#!/bin/bash
#SBATCH --job-name=llm-debate-9b-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_judge_%j.log
#SBATCH --error=logs/debate_9b_judge_%j.err
#
# Experiment 3/4: Pool B (8-9B class) + judge (deepseek-coder-v2:16b)
# ─────────────────────────────────────────────────────────────────
# Peers: granite-code:8b + codegeex4:9b + yi-coder:9b
# Judge: deepseek-coder-v2:16b (MoE, 3B active params, ~9 GB VRAM)
# Features: adaptive_temperature ON, critique_history ON,
# revision_strategy=uniform, show_all_solutions=OFF (best only)
#
# Headline configuration — strongest peers + strongest practical judge.
# Verified locally: this combination achieved 100% on schema_validator and
# mini_database (extreme tasks) in earlier transcripts.
#
# VRAM math on V100 (32 GB): peers ≈ 4.6+5.5+5 = 15 GB, judge ≈ 9 GB
# → 24 GB total (fits, slight headroom for kv-cache).
#
# Usage: sbatch hpc/run_9b_judge.sh

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE="deepseek-coder-v2:16b"
EXP_TAG="9b_judge"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
