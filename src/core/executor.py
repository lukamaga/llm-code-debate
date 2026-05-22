from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Solution, Task, ExecutionResult, CodeQualityMetrics

logger = logging.getLogger(__name__)


class CodeExecutor:

    MAX_CACHE_SIZE = 2048

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self._cache: dict[str, "ExecutionResult"] = {}

    @staticmethod
    def _fix_relative_imports(code: str) -> str:
        import re as _re
        fixed = _re.sub(r'^(from\s+)\.(\w)', r'\1\2', code, flags=_re.MULTILINE)
        if fixed != code:
            logger.debug("Fixed relative imports in solution code")
        return fixed

    @staticmethod
    def _strip_test_functions(code: str) -> str:
        import ast as _ast
        import re as _re

        try:
            tree = _ast.parse(code)
            lines = code.split("\n")
            remove_ranges: list[tuple[int, int]] = []

            for node in tree.body:
                if isinstance(node, _ast.FunctionDef) and node.name.startswith("test_"):
                    start = node.lineno - 1
                    end = node.end_lineno
                    remove_ranges.append((start, end))

            if not remove_ranges:
                return code

            for start, end in reversed(remove_ranges):
                lines[start:end] = []

            result = "\n".join(lines).strip()
            logger.debug(f"Stripped {len(remove_ranges)} test function(s) from solution code")
            return result

        except SyntaxError:
            cleaned_lines = []
            skip = False
            for line in code.split("\n"):
                if _re.match(r"^def test_\w+\s*\(", line):
                    skip = True
                    continue
                if skip:
                    if line and not line[0].isspace():
                        skip = False
                    else:
                        continue
                if not skip:
                    cleaned_lines.append(line)

            return "\n".join(cleaned_lines).strip()

    async def execute(
        self,
        solution: "Solution",
        task: "Task",
    ) -> "ExecutionResult":
        from ..models import ExecutionResult, SolutionStatus, TestResult
        import json as _json

        is_multi_file = getattr(task, 'is_multi_file', False) and solution.code_files

        if is_multi_file:
            code_for_hash = _json.dumps(
                {k: v for k, v in sorted(solution.code_files.items())},
                sort_keys=True,
            )
        else:
            code_for_hash = solution.extract_code_block()
        test_code = "\n".join(task.tests)
        cache_key = hashlib.md5((code_for_hash + test_code).encode()).hexdigest()

        if cache_key in self._cache:
            logger.debug(f"Cache hit for solution {solution.id}")
            return self._cache[cache_key]

        with tempfile.TemporaryDirectory() as tmpdir:
            if is_multi_file:
                extracted = solution.extract_code_files()
                for filename, file_code in extracted.items():
                    file_code = self._strip_test_functions(file_code)
                    file_code = self._fix_relative_imports(file_code)
                    (Path(tmpdir) / filename).write_text(file_code)
                    logger.debug("Multi-file: wrote %s (%d chars)", filename, len(file_code))

                if hasattr(task, 'helper_code') and task.helper_code:
                    for filename, stub_code in task.helper_code.items():
                        stub_path = Path(tmpdir) / filename
                        if not stub_path.exists():
                            stub_path.write_text(stub_code)

                import_lines = "\n".join(
                    getattr(task, 'test_imports', [])
                ) if getattr(task, 'test_imports', []) else ""
                test_content = f"""
import sys
sys.path.insert(0, "{tmpdir}")
{import_lines}

{test_code}
"""
            else:
                code = solution.extract_code_block()

                if hasattr(task, 'helper_code') and task.helper_code:
                    for filename, file_code in task.helper_code.items():
                        helper_path = Path(tmpdir) / filename
                        helper_path.write_text(file_code)

                code = self._strip_test_functions(code)
                solution_path = Path(tmpdir) / "solution.py"
                solution_path.write_text(code)

                import_block = (
                    "import importlib, sys\n"
                    f'sys.path.insert(0, "{tmpdir}")\n'
                    "try:\n"
                    "    _sol = importlib.import_module('solution')\n"
                    "    for _name in dir(_sol):\n"
                    "        if not _name.startswith('test_') and not _name.startswith('_'):\n"
                    "            globals()[_name] = getattr(_sol, _name)\n"
                    "except (ImportError, SyntaxError):\n"
                    "    pass  # let tests fail naturally with NameError\n"
                )

                test_content = f"""
{import_block}

{test_code}
"""
            test_path = Path(tmpdir) / "test_solution.py"
            test_path.write_text(test_content)

            try:
                result = await asyncio.wait_for(
                    self._run_pytest(test_path, tmpdir),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                result = ExecutionResult(
                    status=SolutionStatus.TIMEOUT,
                    tests_passed=0,
                    tests_total=len(task.tests),
                    error_message=f"Execution timed out after {self.timeout}s",
                    execution_time=float(self.timeout),
                )
            except Exception as e:
                result = ExecutionResult(
                    status=SolutionStatus.RUNTIME_ERROR,
                    tests_passed=0,
                    tests_total=len(task.tests),
                    error_message=str(e),
                    execution_time=0.0,
                )

        if len(self._cache) >= self.MAX_CACHE_SIZE:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result
        return result

    async def _run_pytest(
        self,
        test_path: Path,
        working_dir: str,
    ) -> "ExecutionResult":
        from ..models import ExecutionResult, SolutionStatus, TestResult
        import time

        start_time = time.time()

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short",
            "-o", "python_files=test_solution.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        stdout, stderr = await proc.communicate()
        execution_time = time.time() - start_time

        output = stdout.decode() + stderr.decode()

        import re as _re
        tests_passed = 0
        tests_total = 0
        test_results: list[TestResult] = []

        for line in output.split("\n"):
            m = _re.match(r".*::(\w+)\s+(PASSED|FAILED|ERROR)", line)
            if m:
                test_name = m.group(1)
                status_str = m.group(2)
                tests_total += 1
                if status_str == "PASSED":
                    tests_passed += 1
                    test_results.append(TestResult(
                        test_name=test_name,
                        passed=True,
                    ))
                else:
                    test_results.append(TestResult(
                        test_name=test_name,
                        passed=False,
                        error_message=line.strip(),
                    ))

        if proc.returncode == 0:
            status = SolutionStatus.PASSED
        elif "SyntaxError" in output:
            status = SolutionStatus.SYNTAX_ERROR
            logger.warning("SyntaxError in solution: %s", output[:300])
        elif tests_total == 0:
            status = SolutionStatus.RUNTIME_ERROR
            logger.warning(
                "No tests collected (likely ImportError or SyntaxError at module level). "
                "Pytest output: %s", output[:500]
            )
        else:
            status = SolutionStatus.TEST_FAILED

        return ExecutionResult(
            status=status,
            tests_passed=tests_passed,
            tests_total=max(tests_total, 1),
            test_results=test_results,
            execution_time=execution_time,
            error_message=output[:500] if status != SolutionStatus.PASSED else None,
        )


class CodeQualityAnalyzer:

    async def analyze(self, code: str) -> "CodeQualityMetrics":
        from ..models import CodeQualityMetrics

        metrics = CodeQualityMetrics()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            pylint_result = await self._run_pylint(temp_path)
            metrics.pylint_score = pylint_result.get("score", 0.0)
            metrics.pylint_errors = pylint_result.get("errors", 0)
            metrics.pylint_warnings = pylint_result.get("warnings", 0)
            metrics.pylint_conventions = pylint_result.get("conventions", 0)

            radon_result = await self._run_radon(temp_path)
            metrics.cyclomatic_complexity = radon_result.get("avg_complexity", 0.0)
            metrics.max_complexity = radon_result.get("max_complexity", 0)
            metrics.maintainability_index = radon_result.get("mi", 100.0)

            lines = code.split("\n")
            metrics.lines_of_code = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
            metrics.blank_lines = len([l for l in lines if not l.strip()])
            metrics.comment_lines = len([l for l in lines if l.strip().startswith("#")])

        finally:
            os.unlink(temp_path)

        return metrics

    async def _run_pylint(self, filepath: str) -> dict:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-m", "pylint", filepath,
                "--output-format=json",
                "--disable=C0114,C0115,C0116",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            import json
            messages = json.loads(stdout.decode()) if stdout else []

            errors = sum(1 for m in messages if m.get("type") == "error")
            warnings = sum(1 for m in messages if m.get("type") == "warning")
            conventions = sum(1 for m in messages if m.get("type") == "convention")

            total_issues = len(messages)
            score = max(0, 10 - (total_issues * 0.5))

            return {
                "score": score,
                "errors": errors,
                "warnings": warnings,
                "conventions": conventions,
            }
        except Exception as e:
            logger.warning(f"Pylint failed: {e}")
            return {"score": 5.0, "errors": 0, "warnings": 0, "conventions": 0}

    async def _run_radon(self, filepath: str) -> dict:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-m", "radon", "cc", filepath, "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            import json
            cc_data = json.loads(stdout.decode()) if stdout else {}

            complexities = []
            for file_data in cc_data.values():
                if not isinstance(file_data, list):
                    continue
                for item in file_data:
                    complexities.append(item.get("complexity", 1))

            avg_complexity = sum(complexities) / len(complexities) if complexities else 1.0
            max_complexity = max(complexities) if complexities else 1

            proc2 = await asyncio.create_subprocess_exec(
                "python3", "-m", "radon", "mi", filepath, "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await proc2.communicate()

            mi_data = json.loads(stdout2.decode()) if stdout2 else {}
            mi_val = list(mi_data.values())[0] if mi_data else None
            mi = mi_val.get("mi", 100.0) if isinstance(mi_val, dict) else (mi_val if isinstance(mi_val, (int, float)) else 100.0)

            return {
                "avg_complexity": avg_complexity,
                "max_complexity": max_complexity,
                "mi": mi,
            }
        except Exception as e:
            logger.warning(f"Radon failed: {e}")
            return {"avg_complexity": 1.0, "max_complexity": 1, "mi": 100.0}
