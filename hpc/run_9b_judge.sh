#!/bin/bash
#SBATCH --job-name=llm-debate-9b-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_judge_%j.log
#SBATCH --error=logs/debate_9b_judge_%j.err

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE="deepseek-coder-v2:16b"
EXP_TAG="9b_judge"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
