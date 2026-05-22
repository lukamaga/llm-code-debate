#!/bin/bash
#SBATCH --job-name=llm-debate-web
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/web_%j.log
#SBATCH --error=logs/web_%j.err

set -euo pipefail

PROJECT_DIR="/scratch/lustre/home/${USER}/llm-code-debate"
OLLAMA_SIF="${PROJECT_DIR}/hpc/ollama_latest.sif"
OLLAMA_DATA="${PROJECT_DIR}/hpc/ollama_data"
VENV_DIR="${PROJECT_DIR}/venv_hpc"
WEB_PORT=5050

MODELS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct mistral:7b"

cd "${PROJECT_DIR}"
mkdir -p logs results "${OLLAMA_DATA}"

NODE=$(hostname)

echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${NODE}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "============================================"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  To access the web interface, run on your Mac:             ║"
echo "║                                                            ║"
echo "║  ssh -N -L ${WEB_PORT}:${NODE}:${WEB_PORT} ${USER}@hpc.mif.vu.lt  "
echo "║                                                            ║"
echo "║  Then open: http://localhost:${WEB_PORT}                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "${OLLAMA_SIF}" ]; then
    echo "ERROR: Ollama image not found at ${OLLAMA_SIF}"
    echo "Run: cd ${PROJECT_DIR}/hpc && singularity pull ollama_latest.sif docker://ollama/ollama"
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --quiet -r requirements.txt
else
    source "${VENV_DIR}/bin/activate"
fi

echo "Starting Ollama server..."
export OLLAMA_MODELS="${OLLAMA_DATA}"
export OLLAMA_MAX_LOADED_MODELS=5
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

for model in ${MODELS}; do
    echo "Pulling model: ${model}"
    singularity exec \
        --bind "${OLLAMA_DATA}:${OLLAMA_DATA}" \
        "${OLLAMA_SIF}" ollama pull "${model}"
done

export OLLAMA_HOST="http://localhost:11434"

echo ""
echo "Starting web server on port ${WEB_PORT}..."
echo "Press Ctrl+C or scancel ${SLURM_JOB_ID} to stop."
echo ""

cleanup() {
    echo "Shutting down..."
    kill "${OLLAMA_PID}" 2>/dev/null
    wait "${OLLAMA_PID}" 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

python3 -m src.main --web
