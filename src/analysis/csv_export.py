"""
Standalone CSV export of debate results from the SQLite database.

This module mirrors the metric-extraction and CSV-writing logic that the web
``/api/export/csv`` endpoint provides, but does NOT depend on Flask. It is
designed to be invoked from the HPC batch script directly:

    python3 -m src.analysis.csv_export \\
        --db debate_results.db \\
        --out results/summary_<JOB_ID>.csv

The CSV format is identical to the one produced by the web endpoint — same
39 columns, same metric definitions — so spreadsheets generated locally and
on HPC are interchangeable.

The metric-extraction code is intentionally self-contained (not imported
from ``src.web.app``) so the script has zero web/Flask dependencies and
runs on any system with just the project's core requirements.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_metrics_from_record(record) -> dict:
    """Compute extended per-debate metrics from a ``DebateRecord``.

    Mirrors the implementation in ``src/web/app.py::_extract_metrics_from_record``.
    Kept as a separate copy here to avoid pulling Flask into the HPC pipeline.
    Both implementations must be kept in sync — see the test in
    ``tests/test_csv_export.py`` (if added) for parity verification.
    """
    from .metrics_collector import compute_pass_at_k

    data = record.full_debate_data or {}
    rounds = data.get("rounds", [])
    agents = data.get("agents", [])

    # Collect all solutions across rounds
    all_solutions = []
    for rd in rounds:
        for sol in rd.get("solutions", []):
            all_solutions.append(sol)

    n = len(all_solutions)
    c = sum(1 for s in all_solutions
            if s.get("execution_result", {}).get("status") == "passed")

    pass_at_1 = compute_pass_at_k(n, c, 1) if n >= 1 else 0.0
    pass_at_3 = compute_pass_at_k(n, c, 3) if n >= 3 else 0.0

    # Improvement over initial (round 1)
    initial_rates = []
    if rounds:
        for sol in rounds[0].get("solutions", []):
            er = sol.get("execution_result", {})
            initial_rates.append(er.get("pass_rate", 0.0))

    final_rate = record.final_pass_rate or 0.0
    best_initial = max(initial_rates) if initial_rates else 0.0
    avg_initial = (sum(initial_rates) / len(initial_rates)) if initial_rates else 0.0
    if best_initial > 0:
        imp_best = (final_rate - best_initial) / best_initial
    elif final_rate > 0:
        imp_best = final_rate
    else:
        imp_best = 0.0
    if avg_initial > 0:
        imp_avg = (final_rate - avg_initial) / avg_initial
    elif final_rate > 0:
        imp_avg = final_rate
    else:
        imp_avg = 0.0

    # Critique stats
    all_critiques = []
    for rd in rounds:
        all_critiques.extend(rd.get("critiques", []))

    total_critiques = len(all_critiques)
    total_bugs = sum(len(cr.get("bugs", [])) for cr in all_critiques)
    total_improvements = sum(len(cr.get("improvements", [])) for cr in all_critiques)

    # Bug fix rate: tests gained between first and last round
    bugs_fixed = 0
    if len(rounds) >= 2:
        first_sols = {s.get("agent_id"): s for s in rounds[0].get("solutions", [])}
        last_sols = {s.get("agent_id"): s for s in rounds[-1].get("solutions", [])}
        for agent_id, sol1 in first_sols.items():
            sol2 = last_sols.get(agent_id)
            if sol1 and sol2:
                t1 = sol1.get("execution_result", {}).get("tests_passed", 0)
                t2 = sol2.get("execution_result", {}).get("tests_passed", 0)
                if t2 > t1:
                    bugs_fixed += (t2 - t1)
    bug_fix_rate = (bugs_fixed / total_bugs) if total_bugs > 0 else 0.0

    parsed_critiques = [cr for cr in all_critiques if cr.get("ratings_parsed", False)]
    avg_corr = (sum(cr.get("correctness_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0
    avg_eff = (sum(cr.get("efficiency_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0
    avg_read = (sum(cr.get("readability_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0

    final_sol = data.get("final_solution", {})
    final_qm = final_sol.get("quality_metrics", {}) if final_sol else {}
    final_pylint = final_qm.get("pylint_score", 0.0)
    final_complexity = final_qm.get("cyclomatic_complexity", 0.0)

    initial_pylints = []
    initial_complexities = []
    if rounds:
        for sol in rounds[0].get("solutions", []):
            qm = sol.get("quality_metrics", {})
            if qm:
                initial_pylints.append(qm.get("pylint_score", 0.0))
                initial_complexities.append(qm.get("cyclomatic_complexity", 0.0))
    init_avg_pylint = (sum(initial_pylints) / len(initial_pylints)) if initial_pylints else 0.0
    init_avg_complexity = (sum(initial_complexities) / len(initial_complexities)) if initial_complexities else 0.0

    total_llm_time = sum(a.get("stats", a).get("total_generation_time", 0.0) for a in agents)
    total_exec_time = sum(
        s.get("execution_result", {}).get("execution_time", 0.0)
        for s in all_solutions
    )

    final_consensus = data.get("final_consensus", {})
    consensus_ratio = final_consensus.get("consensus_ratio", record.consensus_ratio or 0.0) if final_consensus else (record.consensus_ratio or 0.0)

    rounds_to_consensus = 0
    for rd in rounds:
        cr = rd.get("consensus_result", {})
        if cr and cr.get("reached"):
            rounds_to_consensus = rd.get("round_num", 0)
            break

    # Peak round detection — at which round did the BEST solution appear?
    # This is the core "did debate help?" metric for thesis analysis:
    #   best_round=1 → solved on first try, debate added no value (early stop)
    #   best_round=2+ → cross-pollination through critique/revise actually helped
    # We pick the EARLIEST round that hit the global peak (ties → smaller round
    # number, since reaching peak earlier means less debate effort needed).
    best_round = 1
    best_round_pass_rate = 0.0
    for rd in rounds:
        rd_best = rd.get("best_pass_rate", 0.0) or 0.0
        if rd_best > best_round_pass_rate:
            best_round_pass_rate = rd_best
            best_round = rd.get("round_num", 0)
    peak_after_debate = best_round > 1

    total_rounds = record.total_rounds or len(rounds)
    duration = record.duration_seconds or 0.0
    avg_round_dur = (duration / total_rounds) if total_rounds > 0 else 0.0

    agent_behavior = []
    for a in agents:
        stats = a.get("stats", a)
        agent_behavior.append({
            "agent_id": a.get("id", stats.get("agent_id", "")),
            "model": a.get("model", stats.get("model", "")),
            "critiques_given": stats.get("critiques_given", 0),
            "bugs_found": stats.get("bugs_found", 0),
            "times_changed_mind": stats.get("times_changed_mind", 0),
            "times_defended": stats.get("times_defended", 0),
            "times_adopted_other": stats.get("times_adopted_other", 0),
            "times_won_debate": stats.get("times_won_debate", 0),
        })

    most_bugs_found_by = ""
    most_active_agent = ""
    most_successful_agent = ""
    if agent_behavior:
        by_bugs = max(agent_behavior, key=lambda a: a["bugs_found"])
        if by_bugs["bugs_found"] > 0:
            most_bugs_found_by = by_bugs["agent_id"]
        by_critiques = max(agent_behavior, key=lambda a: a["critiques_given"])
        most_active_agent = by_critiques["agent_id"]
        winners = [a for a in agent_behavior if a["times_won_debate"] > 0]
        if winners:
            most_successful_agent = winners[0]["agent_id"]

    return {
        "pass_at_1": round(pass_at_1, 4),
        "pass_at_3": round(pass_at_3, 4),
        "improvement_over_best_initial": round(imp_best, 4),
        "improvement_over_avg_initial": round(imp_avg, 4),
        "rounds_to_consensus": rounds_to_consensus,
        "total_critiques": total_critiques,
        "total_bugs_found": total_bugs,
        "total_bugs_fixed": bugs_fixed,
        "bug_fix_rate": round(bug_fix_rate, 4),
        "total_improvements_suggested": total_improvements,
        "avg_correctness_rating": round(avg_corr, 2),
        "avg_efficiency_rating": round(avg_eff, 2),
        "avg_readability_rating": round(avg_read, 2),
        "initial_avg_pylint": round(init_avg_pylint, 2),
        "final_pylint": round(final_pylint, 2),
        "initial_avg_complexity": round(init_avg_complexity, 2),
        "final_complexity": round(final_complexity, 2),
        "avg_round_duration": round(avg_round_dur, 2),
        "total_llm_time": round(total_llm_time, 2),
        "total_execution_time": round(total_exec_time, 2),
        "all_solutions_count": n,
        "passing_solutions_count": c,
        "most_active_agent": most_active_agent,
        "most_successful_agent": most_successful_agent,
        "most_bugs_found_by": most_bugs_found_by,
        "consensus_ratio": consensus_ratio,
        "best_round": best_round,
        "best_round_pass_rate": round(best_round_pass_rate, 4),
        "peak_after_debate": peak_after_debate,
    }


# Header order is locked here so HPC- and web-generated CSVs are byte-compatible.
CSV_HEADER = [
    "debate_id", "task_id", "task_name", "difficulty", "mode",
    "agent_models", "num_agents",
    "final_pass_rate", "tests_passed", "tests_total",
    "pass_at_1", "pass_at_3",
    "improvement_over_best_initial", "improvement_over_avg_initial",
    "total_rounds", "consensus_reached", "consensus_ratio", "rounds_to_consensus",
    "total_critiques", "total_bugs_found", "total_bugs_fixed", "bug_fix_rate",
    "total_improvements_suggested",
    "avg_correctness_rating", "avg_efficiency_rating", "avg_readability_rating",
    "initial_avg_pylint", "final_pylint",
    "initial_avg_complexity", "final_complexity",
    "duration_seconds", "avg_round_duration", "total_llm_time", "total_execution_time",
    "all_solutions_count", "passing_solutions_count",
    "most_active_agent", "most_successful_agent", "most_bugs_found_by",
    "status", "winning_agent",
    # Peak-round metrics — answer "did debate help, or solved on round 1?":
    #   best_round=1 + peak_after_debate=False → trivial / early-stop case
    #   best_round=N>1 + peak_after_debate=True → genuine MAD success
    "best_round", "best_round_pass_rate", "peak_after_debate",
]


def export_to_csv(db_path: str, output_path: str) -> int:
    """Read every debate from ``db_path`` and write a CSV summary to
    ``output_path``. Returns the number of rows written (excluding header).

    Creates the parent directory if needed. Overwrites the file if it
    already exists — the HPC script uses a unique filename with the SLURM
    job id, so this only affects deliberate re-runs.
    """
    from ..database.repository import DebateRepository
    from ..database.models import DebateRecord

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    repo = DebateRepository(db_path)
    session = repo._get_session()
    try:
        records = session.query(DebateRecord).order_by(DebateRecord.start_time.asc()).all()
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for r in records:
                mode = "solo" if r.is_solo else "debate"
                models = ",".join(r.agent_models) if r.agent_models else ""
                num_agents = len(r.agent_models) if r.agent_models else 0
                ext = extract_metrics_from_record(r)
                # Prefer the record's stored consensus_ratio for column
                # consistency; ext['consensus_ratio'] is identical when
                # full_debate_data is well-formed but falls back gracefully.
                consensus_ratio = r.consensus_ratio if r.consensus_ratio is not None else ext["consensus_ratio"]
                writer.writerow([
                    r.id, r.task_id, r.task_name, r.task_difficulty, mode,
                    models, num_agents,
                    r.final_pass_rate, r.tests_passed, r.tests_total,
                    ext["pass_at_1"], ext["pass_at_3"],
                    ext["improvement_over_best_initial"], ext["improvement_over_avg_initial"],
                    r.total_rounds, r.consensus_reached, consensus_ratio, ext["rounds_to_consensus"],
                    ext["total_critiques"], ext["total_bugs_found"], ext["total_bugs_fixed"], ext["bug_fix_rate"],
                    ext["total_improvements_suggested"],
                    ext["avg_correctness_rating"], ext["avg_efficiency_rating"], ext["avg_readability_rating"],
                    ext["initial_avg_pylint"], ext["final_pylint"],
                    ext["initial_avg_complexity"], ext["final_complexity"],
                    r.duration_seconds, ext["avg_round_duration"], ext["total_llm_time"], ext["total_execution_time"],
                    ext["all_solutions_count"], ext["passing_solutions_count"],
                    ext["most_active_agent"], ext["most_successful_agent"], ext["most_bugs_found_by"],
                    r.status, r.winning_agent_id,
                    ext["best_round"], ext["best_round_pass_rate"], ext["peak_after_debate"],
                ])
        return len(records)
    finally:
        session.close()


PER_ROUND_HEADER = [
    # Identity
    "debate_id", "task_id", "task_name", "difficulty",
    "agent_id", "model", "role",
    # Round position
    "round_num", "is_revision",
    # Per-round result — round_num=1 is the SINGLE-AGENT BASELINE (the
    # proposal prompt receives only the task, no info from other agents).
    "pass_rate", "tests_passed", "tests_total", "status",
    # Costs
    "generation_time", "code_chars",
    # Flags surfaced by orchestrator's defenses
    "was_truncated", "is_historical_best_reuse",
]


def export_per_round_csv(db_path: str, output_path: str) -> int:
    """Export per-round per-agent solutions to a long-format CSV.

    One row per (debate × agent × round) — i.e. one row per Solution object.
    Round 1 rows give the **single-agent baseline** (proposal phase, no peer
    info), so this CSV alone answers "how much does debate add over solo?"
    without needing a separate solo experiment.

    For a typical batch of 60 tasks × 3 agents × 5 rounds = up to ~900 rows.
    Pivot to wide via Pandas:
        df.pivot_table(index=['debate_id','model'], columns='round_num',
                       values='pass_rate')
    """
    from ..database.repository import DebateRepository
    from ..database.models import DebateRecord

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    repo = DebateRepository(db_path)
    session = repo._get_session()
    rows_written = 0
    try:
        records = session.query(DebateRecord).order_by(DebateRecord.start_time.asc()).all()
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(PER_ROUND_HEADER)

            for r in records:
                data = r.full_debate_data or {}
                rounds = data.get("rounds", [])
                agents = data.get("agents", [])

                # Build a quick lookup: agent_id → role (string), model
                agent_meta = {}
                for a in agents:
                    aid = a.get("id") or a.get("stats", {}).get("agent_id", "")
                    agent_meta[aid] = {
                        "model": a.get("model") or a.get("stats", {}).get("model", ""),
                        "role": (a.get("config", {}) or {}).get("role", "general"),
                    }

                for rd in rounds:
                    round_num = rd.get("round_num", 0)
                    for sol in rd.get("solutions", []):
                        agent_id = sol.get("agent_id", "")
                        meta = agent_meta.get(agent_id, {"model": "", "role": "general"})
                        er = sol.get("execution_result") or {}
                        # `pass_rate` is on execution_result in stored data;
                        # solution-level is sometimes None.
                        pass_rate = er.get("pass_rate")
                        if pass_rate is None:
                            tp = er.get("tests_passed", 0) or 0
                            tt = er.get("tests_total", 0) or 0
                            pass_rate = (tp / tt) if tt > 0 else 0.0
                        code = sol.get("code") or ""
                        writer.writerow([
                            r.id,
                            r.task_id, r.task_name, r.task_difficulty,
                            agent_id, meta["model"], meta["role"],
                            round_num, sol.get("is_revision", False),
                            round(pass_rate or 0.0, 4),
                            er.get("tests_passed", 0),
                            er.get("tests_total", 0),
                            er.get("status", ""),
                            round(sol.get("generation_time", 0.0) or 0.0, 2),
                            len(code),
                            sol.get("was_truncated", False),
                            # Heuristic: a revision that exactly equals an
                            # earlier solution code is a historical-best reuse.
                            # Stored as bool when the orchestrator falls back.
                            sol.get("is_historical_reuse", False),
                        ])
                        rows_written += 1
        return rows_written
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export debate_results.db to flat CSV(s). The summary CSV "
                    "(--out) gives one row per debate (44 columns). The "
                    "per-round CSV (--out-rounds, optional) gives one row per "
                    "(debate × agent × round), so round_num=1 directly "
                    "exposes the single-agent baseline without needing a "
                    "separate solo experiment.",
    )
    parser.add_argument(
        "--db", default="debate_results.db",
        help="Path to the SQLite database (default: debate_results.db)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Summary CSV path (one row per debate). Parent dirs created.",
    )
    parser.add_argument(
        "--out-rounds", default=None,
        help="Optional second CSV — long format, one row per agent per round. "
             "Use this to compare round 1 (baseline) vs later rounds.",
    )
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        return 1

    rows = export_to_csv(args.db, args.out)
    print(f"Exported {rows} debates to {args.out}")

    if args.out_rounds:
        round_rows = export_per_round_csv(args.db, args.out_rounds)
        print(f"Exported {round_rows} per-round solution rows to {args.out_rounds}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
