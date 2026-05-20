# LLM Code Debate

**The Impact of Multi-Agent Debate on the Code Generation Effectiveness of Local Large Language Models (LLMs) Across Varying Task Complexities**

Bachelor's thesis project — source code, experiments and web interface.

## About

- **Author:** Lukaš Patrik Magalinski
- **University:** Vilnius University
- **Faculty:** Faculty of Mathematics and Informatics
- **Study programme:** Informatics
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

## License

MIT — see [LICENSE](LICENSE).

---

*Note: in-code comments and docstrings were drafted with the help of an AI assistant for readability and ease of navigation; the design, experiments and analysis are the author's own work.*
