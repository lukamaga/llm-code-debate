#!/bin/bash

set -e

echo "=== LLM Code Debate Setup ==="

python3 --version || { echo "Python 3 is required"; exit 1; }

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if command -v ollama &> /dev/null; then
    echo "Ollama found. Checking models..."

    echo "Pulling recommended models (this may take a while)..."
    ollama pull qwen2.5-coder:7b || echo "Warning: Could not pull qwen2.5-coder:7b"
    ollama pull deepseek-coder:6.7b || echo "Warning: Could not pull deepseek-coder:6.7b"
    ollama pull codellama:7b-instruct || echo "Warning: Could not pull codellama:7b-instruct"
else
    echo "Warning: Ollama not found. Please install from https://ollama.com"
fi

mkdir -p results
mkdir -p visualizations
mkdir -p logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run a debate:"
echo "  python scripts/quick_run.py --task tasks/medium/lru_cache.json"
echo ""
echo "To start the web interface:"
echo "  python -m src.web.app"
echo ""
