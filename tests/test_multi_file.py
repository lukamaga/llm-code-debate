"""
Tests for multi-file task support (extreme difficulty).
"""
import json
import pytest

from src.models import Task, Solution, ExecutionResult, SolutionStatus
from src.core.prompts import (
    build_multi_file_proposal_prompt,
    extract_multi_file_code_from_response,
    format_multi_file_code_display,
)
from src.core.executor import CodeExecutor


# ============================================================================
# Parser Tests
# ============================================================================

class TestExtractMultiFileCode:
    """Tests for extract_multi_file_code_from_response."""

    def test_standard_format(self):
        """# FILE: x.py labels followed by code blocks."""
        response = """Here's my solution:

# FILE: tokenizer.py
```python
class Tokenizer:
    def tokenize(self, text):
        return text.split()
```

# FILE: evaluator.py
```python
from tokenizer import Tokenizer

class Evaluator:
    pass
```
"""
        result = extract_multi_file_code_from_response(
            response, ["tokenizer.py", "evaluator.py"]
        )
        assert "tokenizer.py" in result
        assert "evaluator.py" in result
        assert "class Tokenizer:" in result["tokenizer.py"]
        assert "from tokenizer import Tokenizer" in result["evaluator.py"]

    def test_alt_format_headers(self):
        """### filename.py headers followed by code blocks."""
        response = """
### tokenizer.py
```python
class Tokenizer:
    pass
```

### evaluator.py
```python
class Evaluator:
    pass
```
"""
        result = extract_multi_file_code_from_response(
            response, ["tokenizer.py", "evaluator.py"]
        )
        assert "tokenizer.py" in result
        assert "evaluator.py" in result

    def test_fallback_by_order(self):
        """Unnamed code blocks assigned to required_files in order."""
        response = """
```python
class Storage:
    pass
```

```python
class Database:
    pass
```
"""
        result = extract_multi_file_code_from_response(
            response, ["storage.py", "database.py"]
        )
        assert "storage.py" in result
        assert "database.py" in result
        assert "class Storage:" in result["storage.py"]
        assert "class Database:" in result["database.py"]

    def test_partial_extraction(self):
        """Only some files found — returns partial dict."""
        response = """
# FILE: tokenizer.py
```python
class Tokenizer:
    pass
```
"""
        result = extract_multi_file_code_from_response(
            response, ["tokenizer.py", "evaluator.py"]
        )
        assert "tokenizer.py" in result
        assert "evaluator.py" not in result

    def test_extra_files_ignored(self):
        """Extra files in response don't cause issues."""
        response = """
# FILE: tokenizer.py
```python
class Tokenizer:
    pass
```

# FILE: evaluator.py
```python
class Evaluator:
    pass
```

# FILE: utils.py
```python
def helper():
    pass
```
"""
        result = extract_multi_file_code_from_response(
            response, ["tokenizer.py", "evaluator.py"]
        )
        assert "tokenizer.py" in result
        assert "evaluator.py" in result
        # Extra file is included (not filtered out)
        assert "utils.py" in result


# ============================================================================
# Model Tests
# ============================================================================

class TestTaskMultiFile:
    """Tests for Task multi-file fields."""

    def test_is_multi_file_true(self):
        task = Task(
            id="test", name="Test", difficulty="extreme",
            description="desc", signature="sig", tests=[],
            required_files=["a.py", "b.py"],
        )
        assert task.is_multi_file is True

    def test_is_multi_file_false(self):
        task = Task(
            id="test", name="Test", difficulty="easy",
            description="desc", signature="sig", tests=[],
        )
        assert task.is_multi_file is False

    def test_from_dict_with_required_files(self):
        data = {
            "id": "calc", "name": "Calculator", "difficulty": "extreme",
            "description": "desc", "signature": "sig",
            "tests": ["def test_x(): pass"],
            "required_files": ["tokenizer.py", "evaluator.py"],
            "test_imports": ["from evaluator import Calc"],
        }
        task = Task.from_dict(data)
        assert task.required_files == ["tokenizer.py", "evaluator.py"]
        assert task.test_imports == ["from evaluator import Calc"]
        assert task.is_multi_file is True

    def test_from_dict_without_required_files(self):
        """Old-format JSON without required_files still loads."""
        data = {
            "id": "old", "name": "Old", "difficulty": "easy",
            "description": "desc", "signature": "sig",
            "tests": ["def test_x(): pass"],
        }
        task = Task.from_dict(data)
        assert task.required_files == []
        assert task.test_imports == []
        assert task.is_multi_file is False

    def test_to_dict_roundtrip(self):
        task = Task(
            id="calc", name="Calculator", difficulty="extreme",
            description="desc", signature="sig",
            tests=["def test(): pass"],
            required_files=["a.py", "b.py"],
            test_imports=["from a import X"],
        )
        d = task.to_dict()
        assert d["required_files"] == ["a.py", "b.py"]
        assert d["test_imports"] == ["from a import X"]
        task2 = Task.from_dict(d)
        assert task2.required_files == task.required_files
        assert task2.test_imports == task.test_imports


class TestSolutionMultiFile:
    """Tests for Solution code_files field."""

    def test_extract_code_files_strips_markdown(self):
        sol = Solution(
            id="s1", agent_id="a1", round_num=1, code="",
            code_files={
                "a.py": "```python\nclass A:\n    pass\n```",
                "b.py": "class B:\n    pass",
            },
        )
        files = sol.extract_code_files()
        assert files["a.py"] == "class A:\n    pass"
        assert files["b.py"] == "class B:\n    pass"

    def test_to_dict_includes_code_files(self):
        sol = Solution(
            id="s1", agent_id="a1", round_num=1, code="combined",
            code_files={"a.py": "class A: pass"},
        )
        d = sol.to_dict()
        assert "code_files" in d
        assert d["code_files"] == {"a.py": "class A: pass"}

    def test_to_dict_omits_empty_code_files(self):
        sol = Solution(
            id="s1", agent_id="a1", round_num=1, code="x = 1",
        )
        d = sol.to_dict()
        assert "code_files" not in d


# ============================================================================
# Prompt Tests
# ============================================================================

class TestMultiFilePrompt:
    """Tests for multi-file prompt builder."""

    def test_contains_required_files(self):
        task = Task(
            id="calc", name="Calculator", difficulty="extreme",
            description="Build a calculator", signature="class Calc: ...",
            tests=[], required_files=["tokenizer.py", "evaluator.py"],
            constraints=["Must use tokenizer"],
        )
        prompt = build_multi_file_proposal_prompt(task)
        assert "tokenizer.py" in prompt
        assert "evaluator.py" in prompt
        assert "# FILE:" in prompt

    def test_contains_format_instructions(self):
        task = Task(
            id="calc", name="Calculator", difficulty="extreme",
            description="desc", signature="sig", tests=[],
            required_files=["a.py", "b.py"],
        )
        prompt = build_multi_file_proposal_prompt(task)
        assert "# FILE: filename1.py" in prompt
        assert "ALL required files" in prompt


class TestFormatMultiFileDisplay:
    """Tests for format_multi_file_code_display."""

    def test_multi_file_display(self):
        sol = Solution(
            id="s1", agent_id="a1", round_num=1, code="combined",
            code_files={"a.py": "class A: pass", "b.py": "class B: pass"},
        )
        display = format_multi_file_code_display(sol)
        assert "# FILE: a.py" in display
        assert "# FILE: b.py" in display
        assert "class A: pass" in display

    def test_single_file_display(self):
        sol = Solution(
            id="s1", agent_id="a1", round_num=1,
            code="```python\ndef foo(): pass\n```",
        )
        display = format_multi_file_code_display(sol)
        assert "def foo(): pass" in display
        assert "# FILE:" not in display


# ============================================================================
# Executor Tests
# ============================================================================

class TestMultiFileExecutor:
    """Tests for CodeExecutor with multi-file solutions."""

    @pytest.fixture
    def executor(self):
        return CodeExecutor(timeout=10)

    @pytest.fixture
    def multi_file_task(self):
        return Task(
            id="mf_test", name="Multi File Test", difficulty="extreme",
            description="Test multi-file execution",
            signature="class Helper: ...\nclass Main: ...",
            tests=[
                "def test_main():\n    m = Main()\n    assert m.greet() == 'hello from helper'",
            ],
            required_files=["helper.py", "main.py"],
            test_imports=["from main import Main"],
        )

    async def test_multi_file_execution_passes(self, executor, multi_file_task):
        """Multi-file solution with correct code passes tests."""
        sol = Solution(
            id="s1", agent_id="a1", round_num=1,
            code="# combined",
            code_files={
                "helper.py": "class Helper:\n    def message(self):\n        return 'hello from helper'",
                "main.py": "from helper import Helper\n\nclass Main:\n    def greet(self):\n        return Helper().message()",
            },
        )
        result = await executor.execute(sol, multi_file_task)
        assert result.status == SolutionStatus.PASSED
        assert result.tests_passed == 1

    async def test_multi_file_import_error(self, executor, multi_file_task):
        """Missing file causes import error."""
        sol = Solution(
            id="s1", agent_id="a1", round_num=1,
            code="# combined",
            code_files={
                # Missing helper.py — main.py will fail to import
                "main.py": "from helper import Helper\n\nclass Main:\n    def greet(self):\n        return Helper().message()",
            },
        )
        result = await executor.execute(sol, multi_file_task)
        assert result.status != SolutionStatus.PASSED

    async def test_single_file_unchanged(self, executor):
        """Single-file path still works."""
        task = Task(
            id="sf_test", name="Single File", difficulty="easy",
            description="desc",
            signature="def add(a, b):",
            tests=["def test_add():\n    assert add(1, 2) == 3"],
        )
        sol = Solution(
            id="s1", agent_id="a1", round_num=1,
            code="def add(a, b):\n    return a + b",
        )
        result = await executor.execute(sol, task)
        assert result.status == SolutionStatus.PASSED

    async def test_multi_file_caching(self, executor, multi_file_task):
        """Multi-file solutions are cached correctly."""
        sol = Solution(
            id="s1", agent_id="a1", round_num=1,
            code="# combined",
            code_files={
                "helper.py": "class Helper:\n    def message(self):\n        return 'hello from helper'",
                "main.py": "from helper import Helper\n\nclass Main:\n    def greet(self):\n        return Helper().message()",
            },
        )
        result1 = await executor.execute(sol, multi_file_task)
        result2 = await executor.execute(sol, multi_file_task)
        assert result1 is result2  # Same object from cache


# ============================================================================
# Task JSON Loading Tests
# ============================================================================

class TestExtremeTaskLoading:
    """Test that extreme task JSON files load correctly."""

    @pytest.fixture(params=["calculator", "mini_database", "event_system"])
    def extreme_task(self, request):
        path = f"tasks/extreme/{request.param}.json"
        with open(path) as f:
            data = json.load(f)
        return Task.from_dict(data)

    def test_is_multi_file(self, extreme_task):
        assert extreme_task.is_multi_file is True

    def test_has_required_files(self, extreme_task):
        assert len(extreme_task.required_files) >= 2

    def test_has_test_imports(self, extreme_task):
        assert len(extreme_task.test_imports) >= 1

    def test_no_helper_code(self, extreme_task):
        """Extreme tasks should NOT provide pre-written helper code."""
        assert extreme_task.helper_code == {}
