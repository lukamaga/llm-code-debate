# Results: CSV exports of the HPC experiments

This folder contains the CSV exports of the 15 SLURM jobs run on the VU MIF HPC cluster for the thesis (see the main [README](../README.md#experimental-setup) and [docs/RESULTS.md](../docs/RESULTS.md)). Each job wrote two files at its end with `python3 -m src.analysis.csv_export`:

- `summary_<experiment tag>_<SLURM job id>.csv`: one row per run (44 columns, thesis §2.9.3);
- `per_round_<experiment tag>_<SLURM job id>.csv`: one row per candidate solution per round (17 columns).

## Files are cumulative

All jobs wrote to the same SQLite database, and each export contains **the whole database at the end of that job**, not only the job's own runs. The 11 main-stage summary exports therefore total 6,060 rows (thesis §3.1). The last file, `summary_9b_judge2_no_adaptive_temp_232581.csv`, contains all 1200 run records: 600 main-stage debates, 360 solo runs and 240 adaptive-temperature-off runs.

A job's own runs are the `debate_id`s that are new compared with the previous file in job-id order:

| File suffix | Stage | Proposers | Judge | Task set | Adaptive temperature | New runs | Rows in summary file |
|---|---|---|---|---|---|---|---|
| `7b_no_judge_217629` | main | Pool A | None | `tasks/` | on | 60 | 60 |
| `7b_judge_217665` | main | Pool A | `deepseek-coder-v2:16b` | `tasks/` | on | 60 | 120 |
| `9b_no_judge_217684` | main | Pool B | None | `tasks/` | on | 60 | 180 |
| `9b_judge_217685` | main | Pool B | `deepseek-coder-v2:16b` | `tasks/` | on | 60 | 240 |
| `all6_solo_217797` | main (solo) | 6 models × 60 tasks | None | `tasks/` | n/a | 360 | 600 |
| `7b_no_judge2_217798` | main | Pool A | None | `tasks2/` | on | 60 | 660 |
| `7b_judge2_217799` | main | Pool A | `deepseek-coder-v2:16b` | `tasks2/` | on | 60 | 720 |
| `9b_no_judge2_217800` | main | Pool B | None | `tasks2/` | on | 60 | 780 |
| `9b_judge2_217801` | main | Pool B | `deepseek-coder-v2:16b` | `tasks2/` | on | 60 | 840 |
| `7b_judge2_r1_217802` | main | Pool A | `deepseek-r1:14b` | `tasks2/` | on | 60 | 900 |
| `9b_judge2_r1_217803` | main | Pool B | `deepseek-r1:14b` | `tasks2/` | on | 60 | 960 |
| `7b_no_judge2_no_adaptive_temp_232578` | ablation | Pool A | None | `tasks2/` | off | 60 | 1020 |
| `7b_judge2_no_adaptive_temp_232579` | ablation | Pool A | `deepseek-coder-v2:16b` | `tasks2/` | off | 60 | 1080 |
| `9b_no_judge2_no_adaptive_temp_232580` | ablation | Pool B | None | `tasks2/` | off | 60 | 1140 |
| `9b_judge2_no_adaptive_temp_232581` | ablation | Pool B | `deepseek-coder-v2:16b` | `tasks2/` | off | 60 | 1200 |

Pool A = `qwen2.5-coder:7b`, `deepseek-coder:6.7b`, `codellama:7b-instruct`. Pool B = `granite-code:8b`, `codegeex4:9b`, `yi-coder:9b`. All debate jobs used `--critique-history`, at most 5 rounds (minimum 2), consensus threshold 0.6, `revision_strategy = uniform` and best-only visibility. Whether adaptive temperature was on is not stored in the data; it is known from the job script (`hpc/run_*.sh`).

Splitting the cumulative files into per-job records (pandas):

```python
import glob, re
import pandas as pd

files = sorted(glob.glob("results/summary_*.csv"),
               key=lambda f: int(re.search(r"_(\d+)\.csv$", f).group(1)))
seen, parts = set(), []
for f in files:
    df = pd.read_csv(f)
    tag, job = re.match(r"results/summary_(.+)_(\d+)\.csv$", f).groups()
    parts.append(df[~df["debate_id"].isin(seen)].assign(exp_tag=tag, slurm_job=int(job)))
    seen |= set(df["debate_id"])
runs = pd.concat(parts, ignore_index=True)   # 1200 rows; 60 per debate job, 360 for the solo job
```

## Summary CSV columns

| Column(s) | Meaning |
|---|---|
| `debate_id`, `task_id`, `task_name`, `difficulty` | Identification. Solo run ids start with `solo_`. |
| `mode` | `debate` or `solo` |
| `agent_models`, `num_agents` | Proposer models, followed by the judge if present |
| `final_pass_rate`, `tests_passed`, `tests_total` | Share and number of the task's unit tests passed by the final solution. The final solution is the best candidate found across all rounds by test pass rate (anti-regression policy, thesis §2.4.1). |
| `pass_at_1`, `pass_at_3` | Chen et al. pass@k estimator over **all** candidate solutions stored in the run (all rounds); a candidate counts as correct when all its tests pass |
| `improvement_over_best_initial`, `improvement_over_avg_initial` | Relative change of `final_pass_rate` against the best / mean round-1 pass rate; equal to `final_pass_rate` when the round-1 value is 0 |
| `total_rounds`, `consensus_reached`, `consensus_ratio`, `rounds_to_consensus` | Debate dynamics |
| `total_critiques`, `total_bugs_found`, `total_improvements_suggested` | Number of critiques, and of bug and improvement items in them |
| `total_bugs_fixed` | Sum over proposers of the increase in passed tests between the agent's round-1 and last-round solution (negative changes count as 0). The thesis calls it "bugs fixed"; it is not linked to the reported bugs. |
| `bug_fix_rate` | `total_bugs_fixed / total_bugs_found` |
| `avg_correctness_rating`, `avg_efficiency_rating`, `avg_readability_rating` | Mean parsed critique ratings (1-10) |
| `initial_avg_pylint`, `final_pylint`, `initial_avg_complexity`, `final_complexity` | Pylint score (0-10) and Radon cyclomatic complexity: mean of round-1 proposals vs final solution |
| `duration_seconds`, `avg_round_duration` | Wall-clock time |
| `total_llm_time` | Generation time of proposal and revision calls (critique and vote calls are not timed) |
| `total_execution_time` | Time spent running pytest |
| `all_solutions_count`, `passing_solutions_count` | Candidate solutions stored, and how many passed all tests |
| `most_active_agent`, `most_successful_agent`, `most_bugs_found_by` | Agent summaries |
| `status`, `winning_agent` | `early_stop`, `consensus_reached` or `max_rounds_reached` (solo runs: `early_stop`), and the winning agent |
| `best_round`, `best_round_pass_rate`, `peak_after_debate` | Round in which the best pass rate was first reached; `peak_after_debate` is true when that round is ≥ 2 |

## Per-round CSV columns

One row per candidate solution per round (run × round × proposing agent): `debate_id`, `task_id`, `task_name`, `difficulty`, `agent_id`, `model`, `role`, `round_num`, `is_revision`, `pass_rate`, `tests_passed`, `tests_total`, `status`, `generation_time`, `code_chars`, `was_truncated`, `is_historical_best_reuse`.

Round 1 holds the independent proposals; solo runs have a single round-1 row. Judges never propose, so they have no rows.

## Known caveats

- `is_historical_best_reuse` is always `False` because of an exporter bug (`csv_export.py` reads a key that is never stored). Slots in which an agent's best earlier solution was reused can only be recovered from the full database.
- `was_truncated` is `False` in every row.
- `role` is always `general`.
- When pytest collects no tests (import or syntax error), `tests_total` is 1 and `pass_rate` is 0.
- Each configuration-task pair was run once, with no seeds.
- The full SQLite database of the HPC runs (`debate_results_hpc_v3.db`, about 216 MB, with complete debate transcripts) is not included in this repository; it is available from the author on request.
