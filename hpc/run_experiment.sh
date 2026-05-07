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
# Peer pool — 3 instruct-tuned code models from independent labs (≈8-9B each).
# Architecture diversity matters more than raw size for debate cross-pollination.
# Verified locally: yi+codegeex+granite peers achieve 100% on schema_validator
# (extreme) when paired with a stronger judge.
PEERS="yi-coder:9b codegeex4:9b granite-code:8b"

# Judge — heterogeneous, stronger than peer pool. Independent lab from peers
# (01.AI Yi, Tsinghua GLM, IBM Granite) — gives clean LLM-as-Judge ablation.
# Set to empty string to run without judge (baseline / no-judge ablation).
#
# VRAM math on V100 (32 GB):
#   peers (5+5.5+4.6 = 15 GB) + judge:
#     qwen2.5-coder:32b   = 19 GB → 34 GB total (overflows V100, slow offload)
#     deepseek-coder-v2:16b = 9 GB → 24 GB total ✅ (fits, MoE = fast)
#     "" (no judge)         = 15 GB total ✅ (fastest, baseline)
# DeepSeek-v2 is MoE (3B active params) → 2× faster inference than 32B.
JUDGE="deepseek-coder-v2:16b"

# 5 rounds — gives adaptive temperature and cross-pollination room to converge.
# Use 3 for quick smoke runs; 5+ for thesis-grade final experiments.
MAX_ROUNDS=5
TASK_DIRS="tasks/easy tasks/medium tasks/hard tasks/extreme"

# Combined list of all models to pull (peers + judge if set).
if [ -n "${JUDGE}" ]; then
    MODELS="${PEERS} ${JUDGE}"
else
    MODELS="${PEERS}"
fi

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
for model in ${PEERS}; do
    AGENTS_ARG="${AGENTS_ARG} ${model}"
done

# Build optional --judge argument only if JUDGE is set
JUDGE_ARG=""
if [ -n "${JUDGE}" ]; then
    JUDGE_ARG="--judge ${JUDGE}"
fi

echo ""
echo "Running experiments..."
echo "Peers:      ${PEERS}"
echo "Judge:      ${JUDGE:-<none>}"
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
            ${JUDGE_ARG} \
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

# ── Export aggregated CSV summary ────────────────────────────────────────
# All debates from this batch (and any previous ones in the same DB) are
# flattened into a single CSV with 39 metric columns: pass rates, pass@k,
# improvement deltas, consensus stats, agent behavior, timings. Same format
# as the web /api/export/csv endpoint, so the file works in Excel/Pandas
# without any extra processing.
#
# Filename includes SLURM_JOB_ID so each batch produces its own CSV — no
# overwriting between runs. The DB itself accumulates all rows, so this
# CSV is a snapshot of "everything in DB at end of this job".
CSV_OUT="results/summary_${SLURM_JOB_ID:-local}.csv"
CSV_ROUNDS="results/per_round_${SLURM_JOB_ID:-local}.csv"
echo ""
echo "Exporting aggregated CSV summaries..."
# Two CSVs: summary (1 row per debate) + per-round (1 row per agent per round).
# Per-round CSV is a goldmine — round_num=1 is the single-agent baseline,
# round_num=N>1 is the debate-improved result, all in one file.
if python3 -m src.analysis.csv_export \
        --db debate_results.db \
        --out "${CSV_OUT}" \
        --out-rounds "${CSV_ROUNDS}"; then
    echo "[OK] Summary:   ${PROJECT_DIR}/${CSV_OUT}"
    echo "[OK] Per-round: ${PROJECT_DIR}/${CSV_ROUNDS}"
else
    echo "[WARN] CSV export failed — DB still contains all results, can re-export later"
fi

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
echo "CSV summary:  ${PROJECT_DIR}/${CSV_OUT}"
echo "CSV rounds:   ${PROJECT_DIR}/${CSV_ROUNDS}"
echo "============================================"

# Cleanup is handled by the trap above
echo "Job done."
