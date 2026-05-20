# LLM Code Debate

**The Impact of Multi-Agent Debate on the Code Generation Effectiveness of Local Large Language Models (LLMs) Across Varying Task Complexities**

Bachelor's thesis project — source code, experiments and web interface.

## About

- **Author:** Lukaš Patrik Magalinski
- **University:** Vilnius University
- **Faculty:** Faculty of Mathematics and Informatics
- **Study programme:** Informatika
- **Supervisor:** Prof. Dr. Aistis Raudys
- **Year:** 2026

## Project summary

A multi-agent debate framework where several local LLMs (run via Ollama) propose, critique, revise and vote on solutions to coding tasks. The framework supports an optional judge agent, runs in CLI or web-UI mode, executes generated code in a sandbox (pytest) and stores all runs in SQLite for later analysis. Experiments span easy, medium and hard tasks and compare solo baselines against debate configurations on a local machine and on the VU MIF HPC cluster.

## Layout

- `src/` — core debate orchestrator, LLM clients, web UI, database, analysis
- `tasks/`, `tasks2/` — benchmark tasks (easy / medium / hard / extreme)
- `hpc/` — Singularity / Slurm scripts for the VU MIF HPC cluster
- `scripts/` — runners, validators and analysis helpers
- `tests/` — unit tests
- `results/`, `transcripts/` — outputs from completed runs

## Running locally

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
python -m src.web.app    # web UI on http://localhost:5050
```

CLI usage:

```bash
python scripts/quick_run.py --task tasks/medium/lru_cache.json
```

## License

MIT — see [LICENSE](LICENSE).
