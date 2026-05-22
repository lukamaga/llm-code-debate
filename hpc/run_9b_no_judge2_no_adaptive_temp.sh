#!/bin/bash
#SBATCH --job-name=llm-debate-9b-nojudge2-no-adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_nojudge2_no_adapt_temp_%j.log
#SBATCH --error=logs/debate_9b_nojudge2_no_adapt_temp_%j.err

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE=""
EXP_TAG="9b_no_judge2_no_adaptive_temp"
EXTRA_MAIN_ARGS="--critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
