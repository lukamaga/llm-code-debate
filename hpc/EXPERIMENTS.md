# Experimental plan — bachelor's thesis ablations

This file documents the full experiment matrix for the LLM Code Debate
thesis. The structure is **HPC for headline results, local for ablations**.

## Phase 1 — HPC headline (4 SLURM jobs)

Goal: measure the **effect of peer-pool strength** and the **effect of
adding a judge**. 2 × 2 design.

All four runs use the **maximally-optimized configuration**:
- `adaptive_temperature` = ON
- `critique_history` = ON
- `revision_strategy` = uniform (baseline; "diverse" tested in Phase 2)
- `show_all_solutions` = OFF (best-only; cheaper, avoids broken-peer copy)
- `max_rounds` = 5
- All 60 tasks: 15 easy + 15 medium + 15 hard + 15 extreme

| # | Script | Peers | Judge |
|---|---|---|---|
| 1 | `run_7b_judge.sh` | qwen2.5-coder:7b + deepseek-coder:6.7b + codellama:7b-instruct | deepseek-coder-v2:16b |
| 2 | `run_7b_no_judge.sh` | (same 7B pool) | none |
| 3 | `run_9b_judge.sh` | granite-code:8b + codegeex4:9b + yi-coder:9b | deepseek-coder-v2:16b |
| 4 | `run_9b_no_judge.sh` | (same 9B pool) | none |

**Estimated GPU budget:** ~4-5 hours per job × 4 jobs = ~18-20 GPU-hours
(within MIF's 100h/month quota).

### Run sequentially (SQLite write contention)

```bash
J1=$(sbatch --parsable hpc/run_7b_judge.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 hpc/run_7b_no_judge.sh)
J3=$(sbatch --parsable --dependency=afterok:$J2 hpc/run_9b_judge.sh)
J4=$(sbatch --parsable --dependency=afterok:$J3 hpc/run_9b_no_judge.sh)
echo "Submitted chain: $J1 → $J2 → $J3 → $J4"
```

Outputs per job (where `<JOB_ID>` is the SLURM id):
- `results/summary_<EXP_TAG>_<JOB_ID>.csv` — 60 rows × 44 cols (one per debate)
- `results/per_round_<EXP_TAG>_<JOB_ID>.csv` — ~750 rows × 17 cols (one per agent×round)
- `transcripts/*.txt` — 60 human-readable per-task transcripts
- `logs/debate_<EXP_TAG>_<JOB_ID>.log` — stdout + stderr

### Headline questions Phase 1 answers

1. **Does peer-pool size matter?** Compare runs 1+2 (7B) vs 3+4 (9B) on
   pass-rate by difficulty. Hypothesis: 9B peers > 7B peers, especially on
   hard/extreme.
2. **Does adding a judge help?** Compare runs 1 vs 2 and 3 vs 4. The
   judge contributes a heterogeneous critic + voter from a different lab
   (DeepSeek) — should reduce hallucinated bug reports and tie-break
   votes when peers disagree.
3. **Does the judge help weaker peers more?** Compare delta(1−2) vs
   delta(3−4). If the gap is larger on the 7B pool, that supports
   "weaker peers benefit more from a stronger external evaluator".

## Phase 2 — Local ablation (single feature flips)

**Run after** Phase 1 reveals the best peer pool (likely 9B based on
prior local runs). Take that pool + judge as the baseline, then flip
ONE flag at a time and re-run on a difficulty-balanced subset.

Why "subset"? Each ablation needs ≥3 seeds per task to beat variance
(see `shortest_path` evidence: same task, same models, results varied
25% – 88% across runs). Running 60 tasks × 3 seeds × 5 ablations would
exhaust GPU budget. Use 20 tasks (5 per difficulty) × 3 seeds × 5
ablations ≈ 300 runs ≈ 4-5 hours locally on Mac (or 2 hours on V100).

### The five ablations

| Ablation | Flag flip | Hypothesis to test |
|---|---|---|
| **Baseline** | (max-config from Phase 1) | reference |
| `−adaptive_temperature` | drop the flag | bumping temp to 0.85 may amplify chaos on 7-9B models |
| `−critique_history` | drop the flag | history may cause critics to repeat old bugs instead of finding new |
| `+diverse_strategy` | `--revision-strategy diverse` | DMAD claims diverse strategies help GPT-4; may confuse weak peers |
| `+show_all_solutions` | add the flag | seeing all peers may invite copying broken code (we observed yi-coder R5 = codegeex4 R2 verbatim) |

### How to run a single ablation locally

```bash
# Example: −adaptive_temperature on a hard subset
for task in tasks/hard/*.json; do
    for seed in 42 43 44; do
        python3 -m src.main \
            --task "$task" \
            --agents granite-code:8b codegeex4:9b yi-coder:9b \
            --judge deepseek-coder-v2:16b \
            --max-rounds 5 \
            --critique-history \
            --output "results/ablation_no_adaptive/seed_$seed/"
    done
done
```

Same command **without** `--adaptive-temperature` to ablate it. (Note:
seed support is not yet wired through to Ollama — see "Open work".)

## Phase 3 — Judge model comparison (local, extreme tasks only)

Goal: justify the choice of `deepseek-coder-v2:16b` against alternatives.

| Judge | Type | Size | Hypothesis |
|---|---|---|---|
| `deepseek-coder-v2:16b` | code-specialised, MoE | 16B (3B active) | Phase 1 default |
| `deepseek-r1:14b` | reasoning-specialised | 14B | thinking model — better at multi-step bugs? |
| `qwen2.5-coder:32b` | code-specialised | 32B | larger code model — strict upper bound on V100 |

Run on 15 extreme tasks × 3 judges × 3 seeds = 135 runs. Each ~5 min →
~11 hours; do this overnight locally.

### Open work for Phase 3

Seed propagation to Ollama is not yet implemented in the code path —
each run is currently non-deterministic. Without seeds, "3 runs per
task" is the best-effort variance proxy (different sampling each time).
A proper `--seed` CLI flag could be added to `src/main.py` and threaded
through `OllamaClient.options.seed`; this is a small future patch.

## What thesis claims need each phase

| Claim | Requires |
|---|---|
| "Multi-agent debate helps over single-agent baseline" | Phase 1 per-round CSV (round 1 = solo) |
| "Stronger peer pool yields better debates" | Phase 1 runs 1+2 vs 3+4 |
| "LLM-as-Judge contributes signal" | Phase 1 odd vs even runs |
| "Judge helps weaker peers more" | Phase 1 delta-of-deltas |
| "Adaptive temperature improves convergence" | Phase 2 ablation, with significance test |
| "Critique history reduces redundant bug reports" | Phase 2 + manual transcript inspection |
| "DMAD-style diverse strategies help/hurt 7-9B peers" | Phase 2 (this is a real risk — may show DMAD is GPT-4 specific) |
| "Showing only best solution avoids copy-paste failures" | Phase 2 + manual transcript inspection |
| "Code-specialised judge ≥ reasoning judge on code" | Phase 3 (interesting if false!) |

## Negative results are valid

If Phase 2 shows e.g. `+diverse_strategy` hurts on extreme tasks, that
**is** a thesis finding — DMAD's claim ("diverse strategies always help")
is then specific to GPT-4-class models, not 7-9B open models. Document
and discuss; do not silently drop the feature.

Same for `−critique_history` — if it improves results, the system was
*hurting itself* by including history, and we have evidence to remove it.

## Files in this directory

- `_lib_run.sh` — shared experiment-pipeline library (sourced by wrappers)
- `run_7b_judge.sh` — Phase 1 run 1
- `run_7b_no_judge.sh` — Phase 1 run 2
- `run_9b_judge.sh` — Phase 1 run 3
- `run_9b_no_judge.sh` — Phase 1 run 4
- `run_experiment.sh` — **legacy** standalone reference (Pool B + judge,
  not used in the 2×2 plan; kept for ad-hoc runs)
- `run_web.sh` — interactive web UI on a compute node (not for batch)
- `singularity.def` — optional custom container build (Ollama image is
  used by all batch scripts via `singularity pull`)
- `README.md` — operational HPC guide (login, quotas, troubleshooting)
- `EXPERIMENTS.md` — this file
