#!/bin/bash
#SBATCH --job-name=llm-debate-9b-judge2-r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_9b_judge2_r1_%j.log
#SBATCH --error=logs/debate_9b_judge2_r1_%j.err

PEERS="granite-code:8b codegeex4:9b yi-coder:9b"
JUDGE="deepseek-r1:14b"
EXP_TAG="9b_judge2_r1"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
