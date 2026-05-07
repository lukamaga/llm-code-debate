#!/bin/bash
# ============================================================================
# _lib_run.sh — shared experiment-pipeline library for ablation SLURM scripts.
#
# NOT directly runnable via sbatch. Use the wrapper scripts that source this:
#   run_7b_judge.sh         — Pool A (7B class) + judge
#   run_7b_no_judge.sh      — Pool A baseline (no judge)
#   run_9b_judge.sh         — Pool B (9B class) + judge
#   run_9b_no_judge.sh      — Pool B baseline (no judge)
#
# Each wrapper sets SBATCH directives + these env vars then sources this file:
#
#   PEERS="model1 model2 model3"          (required, space-separated)
#   JUDGE="judge_model_or_empty_string"   (required; "" = no judge)
#   EXP_TAG="7b_judge"                    (required, used in CSV filenames)
#   EXTRA_MAIN_ARGS="--adaptive-temperature --critique-history"  (optional)
#   MAX_ROUNDS=5                          (optional, default 5)
#   TASK_DIRS="tasks/easy tasks/medium tasks/hard tasks/extreme"  (optional)
#
# The legacy hpc/run_experiment.sh is left untouched as a self-contained
# reference for the "Pool B + judge" config.
# ============================================================================

set -euo pipefail

# ── Defaults (only set if wrapper didn't already set them) ───────────────
: "${MAX_ROUNDS:=5}"
: "${TASK_DIRS:=tasks/easy tasks/medium tasks/hard tasks/extreme}"
: "${EXTRA_MAIN_ARGS:=}"

# Validate required vars from the wrapper.
for var in PEERS EXP_TAG; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var must be set by the wrapper script before sourcing _lib_run.sh" >&2
        exit 2
    fi
done
# JUDGE may legitimately be empty (no-judge ablation), but the variable
# itself must be defined so `set -u` doesn't trip.
: "${JUDGE:=}"

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/lustre/home/${USER}/llm-code-debate"
OLLAMA_SIF="${PROJECT_DIR}/hpc/ollama_latest.sif"
OLLAMA_DATA="${PROJECT_DIR}/hpc/ollama_data"
VENV_DIR="${PROJECT_DIR}/venv_hpc"

# Combined list of all models to pull (peers + judge if set).
if [ -n "${JUDGE}" ]; then
    MODELS="${PEERS} ${JUDGE}"
else
    MODELS="${PEERS}"
fi

# ── Setup ────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"
mkdir -p logs results "${OLLAMA_DATA}"

echo "============================================"
echo "Experiment:   ${EXP_TAG}"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:   $(date)"
echo "Project dir:  ${PROJECT_DIR}"
echo "Peers:        ${PEERS}"
echo "Judge:        ${JUDGE:-<none>}"
echo "Extra flags:  ${EXTRA_MAIN_ARGS:-<none>}"
echo "Max rounds:   ${MAX_ROUNDS}"
echo "============================================"

# ── Check Ollama image ───────────────────────────────────────────────────
if [ ! -f "${OLLAMA_SIF}" ]; then
    echo "ERROR: Ollama image not found at ${OLLAMA_SIF}"
    echo "Run: cd ${PROJECT_DIR}/hpc && singularity pull ollama_latest.sif docker://ollama/ollama"
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
export OLLAMA_MAX_LOADED_MODELS=5
# Flash Attention halves the KV-cache memory footprint on a single V100
# (32 GB), giving ~3-5 GB of headroom. Critical for run_9b_judge.sh —
# without FA the math (24 GB weights + ~9 GB KV cache + 2 GB overhead) is
# right at the 32 GB ceiling. With FA the same config drops to ~30 GB.
# Mathematically equivalent to standard attention (same logits, same
# tokens, same outputs) — just uses tiled SRAM kernel for efficiency.
# Supported on V100 since Ollama 0.4 (Sep 2024). No quality impact, no
# debate-logic impact. Tests still 235 passed after this addition.
export OLLAMA_FLASH_ATTENTION=1
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

JUDGE_ARG=""
if [ -n "${JUDGE}" ]; then
    JUDGE_ARG="--judge ${JUDGE}"
fi

echo ""
echo "Running experiments..."
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

        # Note: ${EXTRA_MAIN_ARGS} is intentionally unquoted so it splits
        # into separate flags (e.g. "--adaptive-temperature --critique-history").
        if python3 -m src.main \
            --task "${task_file}" \
            --agents ${AGENTS_ARG} \
            ${JUDGE_ARG} \
            --max-rounds "${MAX_ROUNDS}" \
            ${EXTRA_MAIN_ARGS} \
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

# ── Export aggregated CSV summaries ──────────────────────────────────────
# Filename includes EXP_TAG + SLURM_JOB_ID so every batch produces its own
# pair of CSVs — no overwriting between runs. The DB itself accumulates
# every debate ever run, so the CSVs are snapshots of "everything in DB
# at end of this job".  Filter by `agent_models` column in Pandas to
# isolate this experiment's rows from prior ones in the shared DB.
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
echo "Total tasks:  ${TOTAL}"
echo "Passed:       ${PASSED}"
echo "Failed:       ${FAILED}"
echo "End time:     $(date)"
echo "Results in:   ${PROJECT_DIR}/results/"
echo "Database:     ${PROJECT_DIR}/debate_results.db"
echo "Summary CSV:  ${PROJECT_DIR}/${CSV_OUT}"
echo "Per-round:    ${PROJECT_DIR}/${CSV_ROUNDS}"
echo "============================================"
echo "Job done."
