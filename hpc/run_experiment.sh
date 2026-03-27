#!/bin/bash
#SBATCH --job-name=llm-debate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/debate_%j.log
#SBATCH --error=logs/debate_%j.err
#
# LLM Code Debate — batch experiment on VU MIF HPC
#
# Usage:
#   sbatch hpc/run_experiment.sh
#
# Prerequisites:
#   1. Ollama image: singularity pull ollama_latest.sif docker://ollama/ollama
#   2. Move image: mv ollama_latest.sif hpc/
#   3. Project copied to /scratch/lustre/home/$USER/llm-code-debate/
#   4. See hpc/README.md for full setup instructions

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/lustre/home/${USER}/llm-code-debate"
OLLAMA_SIF="${PROJECT_DIR}/hpc/ollama_latest.sif"
OLLAMA_DATA="${PROJECT_DIR}/hpc/ollama_data"
VENV_DIR="${PROJECT_DIR}/venv_hpc"

# ── Config (edit these) ───────────────────────────────────────────────────
MODELS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
MAX_ROUNDS=3
TASK_DIRS="tasks/easy tasks/medium tasks/hard"

# ── Setup ─────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"
mkdir -p logs results "${OLLAMA_DATA}"

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:   $(date)"
echo "Project dir:  ${PROJECT_DIR}"
echo "============================================"

# ── Check Ollama image ────────────────────────────────────────────────────
if [ ! -f "${OLLAMA_SIF}" ]; then
    echo "ERROR: Ollama image not found at ${OLLAMA_SIF}"
    echo "Run: cd ${PROJECT_DIR}/hpc && singularity pull ollama_latest.sif docker://ollama/ollama"
    exit 1
fi

# ── Create venv if needed ─────────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --quiet -r requirements.txt
else
    source "${VENV_DIR}/bin/activate"
fi

# ── Cleanup on exit/interrupt ─────────────────────────────────────────────
cleanup() {
    echo "Cleaning up..."
    kill "${OLLAMA_PID}" 2>/dev/null
    wait "${OLLAMA_PID}" 2>/dev/null
    echo "Ollama stopped."
}
trap cleanup EXIT INT TERM

# ── Start Ollama server ──────────────────────────────────────────────────
echo "Starting Ollama server..."
# Singularity runs as current user (not root), so we use OLLAMA_MODELS env var
# to store models in our project dir. Bind-mount ensures path is accessible.
export OLLAMA_MODELS="${OLLAMA_DATA}"
export OLLAMA_MAX_LOADED_MODELS=5
singularity run --nv \
    --bind "${OLLAMA_DATA}:${OLLAMA_DATA}" \
    "${OLLAMA_SIF}" serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready (took ${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama failed to start after 30s"
        kill "${OLLAMA_PID}" 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ── Pull models ──────────────────────────────────────────────────────────
for model in ${MODELS}; do
    echo "Pulling model: ${model}"
    singularity exec \
        --bind "${OLLAMA_DATA}:${OLLAMA_DATA}" \
        "${OLLAMA_SIF}" ollama pull "${model}"
done

# ── Run experiments ──────────────────────────────────────────────────────
export OLLAMA_HOST="http://localhost:11434"

AGENTS_ARG=""
for model in ${MODELS}; do
    AGENTS_ARG="${AGENTS_ARG} ${model}"
done

echo ""
echo "Running experiments..."
echo "Models: ${MODELS}"
echo "Max rounds: ${MAX_ROUNDS}"
echo ""

TOTAL=0
PASSED=0
FAILED=0

for task_dir in ${TASK_DIRS}; do
    if [ ! -d "${task_dir}" ]; then
        echo "WARNING: Directory ${task_dir} not found, skipping"
        continue
    fi

    for task_file in "${task_dir}"/*.json; do
        [ -f "${task_file}" ] || continue
        TOTAL=$((TOTAL + 1))
        TASK_NAME=$(python3 -c "import json; print(json.load(open('${task_file}'))['name'])" 2>/dev/null || echo "${task_file}")

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[${TOTAL}] ${TASK_NAME} (${task_file})"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if python3 -m src.main \
            --task "${task_file}" \
            --agents ${AGENTS_ARG} \
            --max-rounds "${MAX_ROUNDS}" \
            --output "results/"; then
            PASSED=$((PASSED + 1))
            echo "[OK] ${TASK_NAME}"
        else
            FAILED=$((FAILED + 1))
            echo "[FAIL] ${TASK_NAME}"
        fi
        echo ""
    done
done

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "Total tasks:  ${TOTAL}"
echo "Passed:       ${PASSED}"
echo "Failed:       ${FAILED}"
echo "End time:     $(date)"
echo "Results in:   ${PROJECT_DIR}/results/"
echo "Database:     ${PROJECT_DIR}/debate_results.db"
echo "============================================"

# Cleanup is handled by the trap above
echo "Job done."
