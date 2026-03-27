"""
Code execution and quality analysis.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Solution, Task, ExecutionResult, CodeQualityMetrics

logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    Executes code solutions and runs tests via pytest.

    Features:
    - Isolated execution in temp directories
    - Timeout handling
    - Result caching by code hash
    """

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self._cache: dict[str, "ExecutionResult"] = {}

    @staticmethod
    def _strip_test_functions(code: str) -> str:
        """Remove top-level test_ functions from solution code.

        LLMs sometimes include test functions in their solutions.
        These get discovered by pytest and inflate test counts.
        """
        import ast as _ast
        import re as _re

        try:
            tree = _ast.parse(code)
            # Find line ranges of top-level def test_*() functions
            lines = code.split("\n")
            remove_ranges: list[tuple[int, int]] = []

            for node in tree.body:
                if isinstance(node, _ast.FunctionDef) and node.name.startswith("test_"):
                    start = node.lineno - 1  # 0-indexed
                    end = node.end_lineno  # end_lineno is 1-indexed, exclusive after slicing
                    remove_ranges.append((start, end))

            if not remove_ranges:
                return code

            # Remove lines in reverse order to preserve indices
            for start, end in reversed(remove_ranges):
                lines[start:end] = []

            result = "\n".join(lines).strip()
            logger.debug(f"Stripped {len(remove_ranges)} test function(s) from solution code")
            return result

        except SyntaxError:
            # AST failed — fallback to regex removal
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
        """
        Execute a solution against task tests.

        Args:
            solution: The solution to test
            task: The task with test cases

        Returns:
            ExecutionResult with test outcomes
        """
        from ..models import ExecutionResult, SolutionStatus, TestResult
        import json as _json

        # Determine if multi-file
        is_multi_file = getattr(task, 'is_multi_file', False) and solution.code_files

        # Check cache
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

        # Create temp directory for execution
        with tempfile.TemporaryDirectory() as tmpdir:
            if is_multi_file:
                # Multi-file: write each file the LLM produced
                extracted = solution.extract_code_files()
                for filename, file_code in extracted.items():
                    file_code = self._strip_test_functions(file_code)
                    (Path(tmpdir) / filename).write_text(file_code)

                # Write helper stubs if provided (don't overwrite LLM files)
                if hasattr(task, 'helper_code') and task.helper_code:
                    for filename, stub_code in task.helper_code.items():
                        stub_path = Path(tmpdir) / filename
                        if not stub_path.exists():
                            stub_path.write_text(stub_code)

                # Build test file with custom imports
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
                # Single-file path (unchanged)
                code = solution.extract_code_block()

                # Write helper files for legacy helper_code support
                if hasattr(task, 'helper_code') and task.helper_code:
                    for filename, file_code in task.helper_code.items():
                        helper_path = Path(tmpdir) / filename
                        helper_path.write_text(file_code)

                # Write solution code (strip any test_ functions LLMs may have added)
                code = self._strip_test_functions(code)
                solution_path = Path(tmpdir) / "solution.py"
                solution_path.write_text(code)

                # Write test file
                test_content = f"""
import sys
sys.path.insert(0, "{tmpdir}")
from solution import *

{test_code}
"""
            test_path = Path(tmpdir) / "test_solution.py"
            test_path.write_text(test_content)

            # Run pytest
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

        # Cache result
        self._cache[cache_key] = result
        return result

    async def _run_pytest(
        self,
        test_path: Path,
        working_dir: str,
    ) -> "ExecutionResult":
        """Run pytest and parse results."""
        from ..models import ExecutionResult, SolutionStatus, TestResult
        import time

        start_time = time.time()

        # Run pytest with JSON output
        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "pytest", str(test_path), "-v", "--tb=short",
            "-o", "python_files=test_solution.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        stdout, stderr = await proc.communicate()
        execution_time = time.time() - start_time

        output = stdout.decode() + stderr.decode()

        # Parse pytest output
        tests_passed = 0
        tests_total = 0
        test_results: list[TestResult] = []

        for line in output.split("\n"):
            if "PASSED" in line:
                tests_passed += 1
                tests_total += 1
                test_results.append(TestResult(
                    test_name=line.split("::")[1].split()[0] if "::" in line else "test",
                    passed=True,
                ))
            elif "FAILED" in line:
                tests_total += 1
                test_results.append(TestResult(
                    test_name=line.split("::")[1].split()[0] if "::" in line else "test",
                    passed=False,
                    error_message=line,
                ))
            elif "ERROR" in line and "test_" in line:
                tests_total += 1
                test_results.append(TestResult(
                    test_name="test",
                    passed=False,
                    error_message=line,
                ))

        # Determine status
        if proc.returncode == 0:
            status = SolutionStatus.PASSED
        elif "SyntaxError" in output:
            status = SolutionStatus.SYNTAX_ERROR
        elif tests_total == 0:
            status = SolutionStatus.RUNTIME_ERROR
        else:
            status = SolutionStatus.TEST_FAILED

        return ExecutionResult(
            status=status,
            tests_passed=tests_passed,
            tests_total=max(tests_total, 1),  # Ensure at least 1 for division
            test_results=test_results,
            execution_time=execution_time,
            error_message=output[:500] if status != SolutionStatus.PASSED else None,
        )


class CodeQualityAnalyzer:
    """
    Analyzes code quality using Pylint and Radon.
    """

    async def analyze(self, code: str) -> "CodeQualityMetrics":
        """
        Analyze code quality.

        Args:
            code: Python source code

        Returns:
            CodeQualityMetrics with analysis results
        """
        from ..models import CodeQualityMetrics

        metrics = CodeQualityMetrics()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            # Run pylint
            pylint_result = await self._run_pylint(temp_path)
            metrics.pylint_score = pylint_result.get("score", 0.0)
            metrics.pylint_errors = pylint_result.get("errors", 0)
            metrics.pylint_warnings = pylint_result.get("warnings", 0)
            metrics.pylint_conventions = pylint_result.get("conventions", 0)

            # Run radon for complexity
            radon_result = await self._run_radon(temp_path)
            metrics.cyclomatic_complexity = radon_result.get("avg_complexity", 0.0)
            metrics.max_complexity = radon_result.get("max_complexity", 0)
            metrics.maintainability_index = radon_result.get("mi", 100.0)

            # Basic metrics
            lines = code.split("\n")
            metrics.lines_of_code = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
            metrics.blank_lines = len([l for l in lines if not l.strip()])
            metrics.comment_lines = len([l for l in lines if l.strip().startswith("#")])

        finally:
            os.unlink(temp_path)

        return metrics

    async def _run_pylint(self, filepath: str) -> dict:
        """Run pylint and parse results."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-m", "pylint", filepath,
                "--output-format=json",
                "--disable=C0114,C0115,C0116",  # Disable docstring warnings
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            import json
            messages = json.loads(stdout.decode()) if stdout else []

            errors = sum(1 for m in messages if m.get("type") == "error")
            warnings = sum(1 for m in messages if m.get("type") == "warning")
            conventions = sum(1 for m in messages if m.get("type") == "convention")

            # Calculate score (10 - penalties)
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
        """Run radon for complexity metrics."""
        try:
            # Cyclomatic complexity
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
                for item in file_data:
                    complexities.append(item.get("complexity", 1))

            avg_complexity = sum(complexities) / len(complexities) if complexities else 1.0
            max_complexity = max(complexities) if complexities else 1

            # Maintainability index
            proc2 = await asyncio.create_subprocess_exec(
                "python3", "-m", "radon", "mi", filepath, "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await proc2.communicate()

            mi_data = json.loads(stdout2.decode()) if stdout2 else {}
            mi = list(mi_data.values())[0].get("mi", 100.0) if mi_data else 100.0

            return {
                "avg_complexity": avg_complexity,
                "max_complexity": max_complexity,
                "mi": mi,
            }
        except Exception as e:
            logger.warning(f"Radon failed: {e}")
            return {"avg_complexity": 1.0, "max_complexity": 1, "mi": 100.0}
