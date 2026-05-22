from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TASKS_ROOT = Path(__file__).parent.parent / "tasks2"


def run_single_file(task: dict, code: str) -> tuple[int, int, str]:
    test_code = "\n".join(task["tests"])
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        (td / "solution.py").write_text(code)
        import_block = (
            "import importlib, sys\n"
            f'sys.path.insert(0, "{tmpdir}")\n'
            "_sol = importlib.import_module('solution')\n"
            "for _name in dir(_sol):\n"
            "    if not _name.startswith('test_') and not _name.startswith('_'):\n"
            "        globals()[_name] = getattr(_sol, _name)\n"
        )
        (td / "test_solution.py").write_text(f"{import_block}\n\n{test_code}\n")
        return _run_pytest(td)


def run_multi_file(task: dict, files: dict) -> tuple[int, int, str]:
    test_code = "\n".join(task["tests"])
    import_lines = "\n".join(task.get("test_imports", []))
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        for fname, fcode in files.items():
            (td / fname).write_text(fcode)
        test_content = (
            "import sys\n"
            f'sys.path.insert(0, "{tmpdir}")\n'
            f"{import_lines}\n\n{test_code}\n"
        )
        (td / "test_solution.py").write_text(test_content)
        return _run_pytest(td)


def _run_pytest(tmpdir: Path) -> tuple[int, int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmpdir / "test_solution.py"),
         "-v", "--tb=short", "-o", "python_files=test_solution.py"],
        capture_output=True, text=True, cwd=str(tmpdir), timeout=60,
    )
    output = proc.stdout + proc.stderr
    import re
    passed = failed = 0
    failed_names: list[str] = []
    for line in output.split("\n"):
        m = re.match(r".*::(\w+)\s+(PASSED|FAILED|ERROR)", line)
        if m:
            if m.group(2) == "PASSED":
                passed += 1
            else:
                failed += 1
                failed_names.append(m.group(1))
    total = passed + failed
    err = ""
    if failed > 0 or total == 0:
        err = output[-2500:]
        if failed_names:
            err = f"FAILED: {failed_names}\n\n" + err
    return passed, total, err


def load_task(diff: str, name: str) -> dict:
    return json.loads((TASKS_ROOT / diff / f"{name}.json").read_text())


def report(name: str, passed: int, total: int, err: str) -> bool:
    expected = len(load_task("hard" if (TASKS_ROOT / "hard" / f"{name}.json").exists() else "extreme", name)["tests"])
    ok = passed == total == expected
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name:35s} {passed}/{total} (expected {expected})")
    if not ok and err:
        print("    ---")
        for ln in err.split("\n")[:25]:
            print(f"    {ln}")
        print("    ---")
    return ok
