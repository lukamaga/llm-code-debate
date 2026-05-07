#!/bin/bash
#SBATCH --job-name=llm-debate-7b-nojudge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_7b_nojudge_%j.log
#SBATCH --error=logs/debate_7b_nojudge_%j.err
#
# Experiment 2/4: Pool A (7B class) — no judge (baseline)
# ─────────────────────────────────────────────────────────────────
# Peers:        qwen2.5-coder:7b + deepseek-coder:6.7b + codellama:7b-instruct
# Judge:        none (LLM-as-Judge ablation: shows pure peer-debate effect)
# Features:     adaptive_temperature ON, critique_history ON,
#               revision_strategy=uniform, show_all_solutions=OFF (best only)
#
# Pair this with run_7b_judge.sh for the "judge contribution" delta on the
# weaker peer pool. If judge improves 7B more than 9B, that's a finding —
# weaker peers benefit more from a stronger external evaluator.
#
# Usage: sbatch hpc/run_7b_no_judge.sh

PEERS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
JUDGE=""
EXP_TAG="7b_no_judge"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
