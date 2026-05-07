#!/bin/bash
#SBATCH --job-name=llm-debate-7b-judge2-r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_7b_judge2_r1_%j.log
#SBATCH --error=logs/debate_7b_judge2_r1_%j.err
#
# Experiment 1/4 (dataset 2, R1 set): Pool A (7B) + REASONING judge
# ─────────────────────────────────────────────────────────────────
# Peers:        qwen2.5-coder:7b + deepseek-coder:6.7b + codellama:7b-instruct
# Judge:        deepseek-r1:14b  (REASONING model, not code-specialized)
# Features:     adaptive_temperature ON, critique_history ON
# Dataset:      tasks2/hard + tasks2/extreme
#
# Research question: does a REASONING-tuned judge (deepseek-r1:14b) improve
# debate outcomes more or less than a CODE-tuned judge (deepseek-coder-v2:16b)?
#
# Comparison:
#   run_7b_judge2.sh       → coder judge (deepseek-coder-v2:16b)
#   run_7b_judge2_r1.sh    → reasoning judge (deepseek-r1:14b)  ← THIS SCRIPT
# Same peers, same dataset, same flags — only the judge differs.
#
# VRAM math on V100 (32 GB):
#   peers ≈ 12.3 GB + r1:14b ≈ 9 GB + KV cache ≈ 5 GB → ~26 GB ✅
#
# Usage: sbatch hpc/run_7b_judge2_r1.sh

PEERS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
JUDGE="deepseek-r1:14b"
EXP_TAG="7b_judge2_r1"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
