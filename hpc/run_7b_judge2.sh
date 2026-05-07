#!/bin/bash
#SBATCH --job-name=llm-debate-7b-judge2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_7b_judge2_%j.log
#SBATCH --error=logs/debate_7b_judge2_%j.err
#
# Experiment 1/4 (dataset 2): Pool A (7B class) + judge (deepseek-coder-v2:16b)
# ─────────────────────────────────────────────────────────────────
# Peers:        qwen2.5-coder:7b + deepseek-coder:6.7b + codellama:7b-instruct
# Judge:        deepseek-coder-v2:16b (MoE, 3B active params, ~9 GB VRAM)
# Features:     adaptive_temperature ON, critique_history ON,
#               revision_strategy=uniform, show_all_solutions=OFF (best only)
# Dataset:      tasks2/hard + tasks2/extreme (NEW second dataset)
#
# VRAM math on V100 (32 GB):  peers ≈ 4.7+3.8+3.8 = 12.3 GB,
# judge ≈ 9 GB → 21 GB total ✅ (comfortable fit, MoE = fast inference).
#
# Usage: sbatch hpc/run_7b_judge2.sh

PEERS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
JUDGE="deepseek-coder-v2:16b"
EXP_TAG="7b_judge2"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
