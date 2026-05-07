#!/bin/bash
#SBATCH --job-name=llm-debate-all6-solo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=logs/debate_all6_solo_%j.log
#SBATCH --error=logs/debate_all6_solo_%j.err
#
# Experiment: SOLO baseline — each of 6 models runs all 60 tasks ALONE
# ─────────────────────────────────────────────────────────────────
# Models (6):   qwen2.5-coder:7b, deepseek-coder:6.7b, codellama:7b-instruct,
#               granite-code:8b, codegeex4:9b, yi-coder:9b
# Judge:        none (solo mode = no debate, no critique, no vote)
# Dataset:      tasks/easy + tasks/medium + tasks/hard + tasks/extreme (60 tasks)
# Total runs:   6 models × 60 tasks = 360 solo runs
#
# Sequence: model 1 does ALL 60 tasks, then model 2 does all 60, etc.
# Each task call uses --solo flag, runs single agent with no debate.
# Records saved to debate_results.db with mode='solo' for filtering.
#
# This is the SOLO BASELINE for the thesis — answers:
# "Does debate (3 or 6 peers) actually beat a single agent?"
#
# VRAM math on V100 (32 GB): only 1 model loaded at a time = ~5 GB peak ✅
#
# Estimated runtime: ~6-8 hours (360 tasks × ~60s avg)
#   Easy/medium tasks early-stop in seconds, hard/extreme take 30-90s each
#
# Usage: sbatch hpc/run_all6_solo.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/lustre/home/${USER}/llm-code-debate"
OLLAMA_SIF="${PROJECT_DIR}/hpc/ollama_latest.sif"
OLLAMA_DATA="${PROJECT_DIR}/hpc/ollama_data"
VENV_DIR="${PROJECT_DIR}/venv_hpc"

# All 6 peer models (no judge, since solo = no critique/vote)
MODELS=(
    "qwen2.5-coder:7b"
    "deepseek-coder:6.7b"
    "codellama:7b-instruct"
    "granite-code:8b"
    "codegeex4:9b"
    "yi-coder:9b"
)

EXP_TAG="all6_solo"
TASK_DIRS="tasks/easy tasks/medium tasks/hard tasks/extreme"

# ── Setup ────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"
mkdir -p logs results "${OLLAMA_DATA}"

echo "============================================"
echo "Experiment:   ${EXP_TAG}"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:   $(date)"
echo "Models:       ${MODELS[*]}"
echo "Total runs:   ${#MODELS[@]} models × tasks in (${TASK_DIRS})"
echo "============================================"

# ── Check Ollama image ───────────────────────────────────────────────────
if [ ! -f "${OLLAMA_SIF}" ]; then
    echo "ERROR: Ollama image not found at ${OLLAMA_SIF}"
    exit 1
fi

# ── Create venv if needed ────────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --quiet -r requirements.txt
else
    source "${VENV_DIR}/bin/activate"
fi

# ── Cleanup on exit/interrupt ────────────────────────────────────────────
cleanup() {
    echo "Cleaning up..."
    if [ -n "${OLLAMA_PID:-}" ]; then
        kill "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
    fi
    echo "Ollama stopped."
}
trap cleanup EXIT INT TERM

# ── Start Ollama server ──────────────────────────────────────────────────
echo "Starting Ollama server..."
export OLLAMA_MODELS="${OLLAMA_DATA}"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1
singularity run --nv \
    --bind "${OLLAMA_DATA}:${OLLAMA_DATA}" \
    "${OLLAMA_SIF}" serve &
OLLAMA_PID=$!

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

# ── Pull all models (skipped if already cached) ──────────────────────────
for model in "${MODELS[@]}"; do
    echo "Ensuring model: ${model}"
    singularity exec \
        --bind "${OLLAMA_DATA}:${OLLAMA_DATA}" \
        "${OLLAMA_SIF}" ollama pull "${model}"
done

# ── Run experiments: each model goes through all tasks alone ─────────────
export OLLAMA_HOST="http://localhost:11434"

TOTAL=0
PASSED=0
FAILED=0

for model in "${MODELS[@]}"; do
    echo ""
    echo "############################################"
    echo "# SOLO RUN: ${model}"
    echo "############################################"
    echo ""

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
            echo "[${TOTAL}] ${model} | ${TASK_NAME}"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            if python3 -m src.main \
                --task "${task_file}" \
                --agents "${model}" \
                --solo \
                --output "results/"; then
                PASSED=$((PASSED + 1))
                echo "[OK] ${model} | ${TASK_NAME}"
            else
                FAILED=$((FAILED + 1))
                echo "[FAIL] ${model} | ${TASK_NAME}"
            fi
            echo ""
        done
    done
done

# ── Export aggregated CSV summaries ──────────────────────────────────────
CSV_OUT="results/summary_${EXP_TAG}_${SLURM_JOB_ID:-local}.csv"
CSV_ROUNDS="results/per_round_${EXP_TAG}_${SLURM_JOB_ID:-local}.csv"
echo ""
echo "Exporting aggregated CSV summaries..."
if python3 -m src.analysis.csv_export \
        --db debate_results.db \
        --out "${CSV_OUT}" \
        --out-rounds "${CSV_ROUNDS}"; then
    echo "[OK] Summary:   ${PROJECT_DIR}/${CSV_OUT}"
    echo "[OK] Per-round: ${PROJECT_DIR}/${CSV_ROUNDS}"
else
    echo "[WARN] CSV export failed — DB still contains all results, can re-export later"
fi

# ── Final summary ────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE: ${EXP_TAG}"
echo "Total runs:   ${TOTAL} (expected: 6 models × 60 tasks = 360)"
echo "Passed:       ${PASSED}"
echo "Failed:       ${FAILED}"
echo "End time:     $(date)"
echo "Results in:   ${PROJECT_DIR}/results/"
echo "Database:     ${PROJECT_DIR}/debate_results.db"
echo "Summary CSV:  ${PROJECT_DIR}/${CSV_OUT}"
echo "Per-round:    ${PROJECT_DIR}/${CSV_ROUNDS}"
echo "============================================"
echo "Job done."
