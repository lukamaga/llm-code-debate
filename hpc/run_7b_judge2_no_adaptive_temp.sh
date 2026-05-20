#!/bin/bash
#SBATCH --job-name=llm-debate-7b-judge2-no-adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_7b_judge2_no_adapt_temp_%j.log
#SBATCH --error=logs/debate_7b_judge2_no_adapt_temp_%j.err
#
# Ablation: Pool A (7B class) + judge (deepseek-coder-v2:16b), NO adaptive_temperature
# ─────────────────────────────────────────────────────────────────
# Peers: qwen2.5-coder:7b + deepseek-coder:6.7b + codellama:7b-instruct
# Judge: deepseek-coder-v2:16b (MoE, 3B active params, ~9 GB VRAM)
# Features: adaptive_temperature OFF ablation target
# critique_history ON,
# revision_strategy=uniform, show_all_solutions=OFF (best only)
# Dataset: tasks2/hard + tasks2/extreme (same as run_7b_judge2.sh)
#
# Purpose: A/B ablation для empirical validation of adaptive_temperature
# mechanism в presence of code-specialised judge. Pair-compare против
# run_7b_judge2.sh (same config, но с --adaptive-temperature ON).
#
# Expected analysis:
# - paired Student's t-test по 60 задачам (tasks2/)
# - difference в avg final_pass_rate (Pool A + kodo: ON vs OFF)
# - interaction analysis: ablation effect стronger без teisėjas или с ним?
# - distribution of stagnant_rounds count (was the mechanism triggered?)
#
# Pair this with run_9b_judge2_no_adaptive_temp.sh + ablation no_judge2
# scripts for full 2x2 factorial (pool × judge) adaptive_temperature ablation.
#
# VRAM math on V100 (32 GB): peers ≈ 4.7+3.8+3.8 = 12.3 GB,
# judge ≈ 9 GB → 21 GB total (same budget as run_7b_judge2.sh).
#
# Usage: sbatch hpc/run_7b_judge2_no_adaptive_temp.sh

PEERS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
JUDGE="deepseek-coder-v2:16b"
EXP_TAG="7b_judge2_no_adaptive_temp"
EXTRA_MAIN_ARGS="--critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
