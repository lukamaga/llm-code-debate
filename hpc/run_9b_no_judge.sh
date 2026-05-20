#!/bin/bash
#SBATCH --job-name=llm-debate-9b-nojudge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_nojudge_%j.log
#SBATCH --error=logs/debate_9b_nojudge_%j.err
#
# Experiment 4/4: Pool B (8-9B class) — no judge (baseline)
# ─────────────────────────────────────────────────────────────────
# Peers: granite-code:8b + codegeex4:9b + yi-coder:9b
# Judge: none (LLM-as-Judge ablation: shows pure peer-debate effect)
# Features: adaptive_temperature ON, critique_history ON,
# revision_strategy=uniform, show_all_solutions=OFF (best only)
#
# Pair this with run_9b_judge.sh for the cleanest LLM-as-Judge ablation on
# the strongest peer pool. Compare per-difficulty: judge effect should be
# largest on extreme tasks where peers struggle to self-correct.
#
# VRAM math on V100 (32 GB): peers ≈ 15 GB total (lots of headroom).
#
# Usage: sbatch hpc/run_9b_no_judge.sh

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE=""
EXP_TAG="9b_no_judge"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
