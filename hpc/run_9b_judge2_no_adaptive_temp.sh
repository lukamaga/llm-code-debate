#!/bin/bash
#SBATCH --job-name=llm-debate-9b-judge2-no-adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_judge2_no_adapt_temp_%j.log
#SBATCH --error=logs/debate_9b_judge2_no_adapt_temp_%j.err
#
# Ablation: Pool B (8-9B class) + judge (deepseek-coder-v2:16b), NO adaptive_temperature
# ─────────────────────────────────────────────────────────────────
# Peers:        granite-code:8b + codegeex4:9b + yi-coder:9b
# Judge:        deepseek-coder-v2:16b (MoE, 3B active params, ~9 GB VRAM)
# Features:     adaptive_temperature OFF  ← ablation target
#               critique_history ON,
#               revision_strategy=uniform, show_all_solutions=OFF (best only)
# Dataset:      tasks2/hard + tasks2/extreme (same as run_9b_judge2.sh)
#
# Purpose: A/B ablation для empirical validation of adaptive_temperature
# mechanism в headline configuration (strongest peers + strongest practical
# judge). Pair-compare против run_9b_judge2.sh (same config, но с
# --adaptive-temperature ON).
#
# Expected analysis:
#   - paired Student's t-test по 60 задачам (tasks2/)
#   - difference в avg final_pass_rate (Pool B + kodo: ON vs OFF)
#   - interaction analysis: помогает ли adaptive_temperature компенсировать
#     kodo teisėjas neigiamą poveikį, kuris buvo užfiksuotas 4.4 poskyryje?
#   - distribution of stagnant_rounds count (was the mechanism triggered?)
#
# Pair this with run_7b_judge2_no_adaptive_temp.sh + ablation no_judge2
# scripts for full 2x2 factorial (pool × judge) adaptive_temperature ablation.
#
# VRAM math on V100 (32 GB):  peers ≈ 4.6+5.5+5 = 15 GB, judge ≈ 9 GB
# → 24 GB total ✅ (same budget as run_9b_judge2.sh).
#
# Usage: sbatch hpc/run_9b_judge2_no_adaptive_temp.sh

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE="deepseek-coder-v2:16b"
EXP_TAG="9b_judge2_no_adaptive_temp"
EXTRA_MAIN_ARGS="--critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
