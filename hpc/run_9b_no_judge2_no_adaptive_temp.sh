#!/bin/bash
#SBATCH --job-name=llm-debate-9b-nojudge2-no-adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_nojudge2_no_adapt_temp_%j.log
#SBATCH --error=logs/debate_9b_nojudge2_no_adapt_temp_%j.err
#
# Ablation: Pool B (8-9B class) — no judge, NO adaptive_temperature
# ─────────────────────────────────────────────────────────────────
# Peers:        granite-code:8b + codegeex4:9b + yi-coder:9b
# Judge:        none (peer-only debate)
# Features:     adaptive_temperature OFF  ← ablation target
#               critique_history ON,
#               revision_strategy=uniform, show_all_solutions=OFF (best only)
# Dataset:      tasks2/hard + tasks2/extreme (same as run_9b_no_judge2.sh)
#
# Purpose: A/B ablation для empirical validation of adaptive_temperature
# mechanism. Pair-compare против run_9b_no_judge2.sh (same config, но с
# --adaptive-temperature ON). Все остальные параметры идентичны.
#
# Expected analysis:
#   - paired Student's t-test по 60 задачам (tasks2/)
#   - difference в avg final_pass_rate (Pool B ON vs OFF)
#   - distribution of stagnant_rounds count (was the mechanism triggered?)
#   - effective temperature trace в logs (для verification)
#
# Pair this with run_7b_no_judge2_no_adaptive_temp.sh for cross-pool sanity
# check — does the effect (если есть) replicate в both 7B and 9B classes?
#
# VRAM math on V100 (32 GB):  peers ≈ 15 GB total ✅ (lots of headroom).
#
# Usage: sbatch hpc/run_9b_no_judge2_no_adaptive_temp.sh

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE=""
EXP_TAG="9b_no_judge2_no_adaptive_temp"
EXTRA_MAIN_ARGS="--critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
