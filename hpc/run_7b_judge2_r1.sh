#!/bin/bash
#SBATCH --job-name=llm-debate-7b-judge2-r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debate_7b_judge2_r1_%j.log
#SBATCH --error=logs/debate_7b_judge2_r1_%j.err

PEERS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
JUDGE="deepseek-r1:14b"
EXP_TAG="7b_judge2_r1"
EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"
MAX_ROUNDS=5
TASK_DIRS="tasks2/hard tasks2/extreme"

source "${SLURM_SUBMIT_DIR}/hpc/_lib_run.sh"
