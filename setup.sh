#!/bin/bash
# Setup script for LLM Code Debate System

set -e

echo "=== LLM Code Debate Setup ==="

# Check Python version
python3 --version || { echo "Python 3 is required"; exit 1; }

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama found. Checking models..."

    # Pull recommended models
    echo "Pulling recommended models (this may take a while)..."
    ollama pull qwen2.5-coder:7b || echo "Warning: Could not pull qwen2.5-coder:7b"
    ollama pull deepseek-coder:6.7b || echo "Warning: Could not pull deepseek-coder:6.7b"
    ollama pull codellama:7b || echo "Warning: Could not pull codellama:7b"
else
    echo "Warning: Ollama not found. Please install from https://ollama.com"
fi

# Create necessary directories
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
