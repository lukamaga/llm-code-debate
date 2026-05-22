from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SolutionStatus(Enum):
    PENDING = "pending"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TEST_FAILED = "test_failed"
    TIMEOUT = "timeout"
    PASSED = "passed"


@dataclass
class TestResult:
    test_name: str
    passed: bool
    execution_time: float = 0.0
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class ExecutionResult:
    status: SolutionStatus
    tests_passed: int = 0
    tests_total: int = 0
    test_results: list[TestResult] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: str | None = None
    
    @property
    def pass_rate(self) -> float:
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total
    
    @property
    def all_passed(self) -> bool:
        return self.status == SolutionStatus.PASSED
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "pass_rate": self.pass_rate,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "test_results": [t.to_dict() for t in self.test_results],
        }


@dataclass
class CodeQualityMetrics:
    pylint_score: float = 0.0
    pylint_errors: int = 0
    pylint_warnings: int = 0
    pylint_conventions: int = 0
    
    cyclomatic_complexity: float = 0.0
    max_complexity: int = 0
    maintainability_index: float = 0.0
    
    lines_of_code: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pylint_score": self.pylint_score,
            "pylint_errors": self.pylint_errors,
            "pylint_warnings": self.pylint_warnings,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_complexity": self.max_complexity,
            "maintainability_index": self.maintainability_index,
            "lines_of_code": self.lines_of_code,
        }


@dataclass
class Solution:
    id: str
    agent_id: str
    round_num: int
    code: str
    
    execution_result: ExecutionResult | None = None
    
    quality_metrics: CodeQualityMetrics | None = None
    
    is_revision: bool = False
    parent_solution_id: str | None = None
    generation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    code_files: dict[str, str] = field(default_factory=dict)

    votes_received: int = 0

    was_truncated: bool = False
    
    @property
    def passed_all_tests(self) -> bool:
        if self.execution_result is None:
            return False
        return self.execution_result.all_passed
    
    @property
    def pass_rate(self) -> float:
        if self.execution_result is None:
            return 0.0
        return self.execution_result.pass_rate
    
    def extract_code_files(self) -> dict[str, str]:
        result = {}
        for filename, file_code in self.code_files.items():
            code = file_code.strip()
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            result[filename] = code.strip()
        return result

    def extract_code_block(self) -> str:
        code = self.code.strip()
        
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "agent_id": self.agent_id,
            "round_num": self.round_num,
            "code": self.code,
            "is_revision": self.is_revision,
            "parent_solution_id": self.parent_solution_id,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp.isoformat(),
            "votes_received": self.votes_received,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "quality_metrics": self.quality_metrics.to_dict() if self.quality_metrics else None,
        }
        if self.code_files:
            d["code_files"] = self.code_files
        if self.was_truncated:
            d["was_truncated"] = True
        return d


@dataclass
class Task:
    id: str
    name: str
    difficulty: str
    description: str
    signature: str
    tests: list[str]
    constraints: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    expected_complexity: str | None = None
    tags: list[str] = field(default_factory=list)
    helper_code: dict[str, str] = field(default_factory=dict)
    required_files: list[str] = field(default_factory=list)
    test_imports: list[str] = field(default_factory=list)

    @property
    def is_multi_file(self) -> bool:
        return len(self.required_files) > 0
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(
            id=data["id"],
            name=data["name"],
            difficulty=data.get("difficulty", "medium"),
            description=data["description"],
            signature=data["signature"],
            tests=data["tests"],
            constraints=data.get("constraints", []),
            hints=data.get("hints", []),
            expected_complexity=data.get("expected_complexity"),
            tags=data.get("tags", []),
            helper_code=data.get("helper_code", {}),
            required_files=data.get("required_files", []),
            test_imports=data.get("test_imports", []),
        )
    
    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "signature": self.signature,
            "tests": self.tests,
            "constraints": self.constraints,
            "hints": self.hints,
            "expected_complexity": self.expected_complexity,
            "tags": self.tags,
        }
        if self.helper_code:
            d["helper_code"] = self.helper_code
        if self.required_files:
            d["required_files"] = self.required_files
        if self.test_imports:
            d["test_imports"] = self.test_imports
        return d
    
    def get_test_code(self) -> str:
        return "\n\n".join(self.tests)
