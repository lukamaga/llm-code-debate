"""Driver for validating tasks2/ via reference solutions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from validate_tasks2 import load_task, run_single_file, run_multi_file


def validate_group(name: str, refs: dict, diff: str) -> list:
    print("="*80)
    print(f"VALIDATING {name}")
    print("="*80)
    failures = []
    for tid, code in refs.items():
        task = load_task(diff, tid)
        if isinstance(code, dict):
            passed, total, err = run_multi_file(task, code)
        else:
            passed, total, err = run_single_file(task, code)
        expected = len(task["tests"])
        ok = passed == total == expected
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {tid:35s} {passed}/{total}/{expected}")
        if not ok:
            failures.append((tid, passed, total, expected, err))
    return failures


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "all"
    all_failures = []

    if group in ("hard", "all"):
        from refs_hard import REFS
        all_failures += validate_group("HARD (5 NEW)", REFS, "hard")

    if group in ("e1", "all"):
        from refs_extreme_1 import REFS
        all_failures += validate_group("EXTREME BATCH 1", REFS, "extreme")

    if group in ("e2", "all"):
        try:
            from refs_extreme_2 import REFS
            all_failures += validate_group("EXTREME BATCH 2", REFS, "extreme")
        except ImportError:
            print("  (refs_extreme_2 not yet defined)")

    if group in ("e3", "all"):
        try:
            from refs_extreme_3 import REFS
            all_failures += validate_group("EXTREME BATCH 3", REFS, "extreme")
        except ImportError:
            print("  (refs_extreme_3 not yet defined)")

    if group in ("e4", "all"):
        try:
            from refs_extreme_4 import REFS
            all_failures += validate_group("EXTREME BATCH 4", REFS, "extreme")
        except ImportError:
            print("  (refs_extreme_4 not yet defined)")

    if group in ("e5", "all"):
        try:
            from refs_extreme_5 import REFS
            all_failures += validate_group("EXTREME BATCH 5", REFS, "extreme")
        except ImportError:
            print("  (refs_extreme_5 not yet defined)")

    if all_failures:
        print("\n" + "="*80)
        print("FAILURES")
        print("="*80)
        for tid, p, t, e, err in all_failures:
            print(f"\n--- {tid}: passed={p}, total={t}, expected={e} ---")
            for ln in err.split("\n")[-25:]:
                print(f"  {ln}")
        sys.exit(1)
    print("\nAll validated tasks PASS.")


if __name__ == "__main__":
    main()
