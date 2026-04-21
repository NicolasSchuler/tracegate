"""Integration tests for TraceCoder's patched exact-test-context flow."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

_TRACECODER_REPO_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "repos" / "TraceCoder"
_TRACECODER_STALE_MODULES = (
    "config",
    "datasets",
    "datasets.llmdebug",
    "datasets.llmdebug.evaluation",
    "problem_processor",
    "reporting",
    "src",
    "src.exact_test_context",
    "src.generation",
    "src.postprocessing",
    "src.progress",
    "src.traceRunner",
    "trace_learn_coder",
)


def _require_tracecoder_repo() -> None:
    if not _TRACECODER_REPO_ROOT.is_dir():
        pytest.skip("real TraceCoder repo not provisioned")


def _import_tracecoder_repo_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    *,
    stub_openai: bool = False,
    stub_pandas: bool = False,
) -> ModuleType:
    _require_tracecoder_repo()
    monkeypatch.syspath_prepend(str(_TRACECODER_REPO_ROOT))
    if stub_openai:
        monkeypatch.setitem(
            sys.modules,
            "openai",
            SimpleNamespace(
                APIError=RuntimeError,
                OpenAI=object,
                RateLimitError=RuntimeError,
                Timeout=RuntimeError,
            ),
        )
    if stub_pandas:
        monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(DataFrame=object))
    for stale_name in (module_name, *_TRACECODER_STALE_MODULES):
        sys.modules.pop(stale_name, None)
    return importlib.import_module(module_name)


class TestExactTestContext:
    def test_small_test_file_returns_full_verbatim(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        test_code = (
            "from buggy import multiply\n\n\n"
            "def test_multiply():\n"
            "    assert multiply(3, 4) == 12\n"
        )

        rendered, meta = build_exact_test_context(
            test_code,
            failed_nodeids=["test_buggy.py::test_multiply"],
            failure_output="",
            max_chars=400,
        )

        assert rendered == test_code.strip()
        assert meta["source"] == "full_file"
        assert meta["chars"] == len(rendered)

    def test_selects_only_matching_pytest_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        test_code = (
            "from buggy import multiply\n\n\n"
            "def test_addition_regression():\n"
            "    assert multiply(3, 4) == 7\n\n\n"
            "def test_multiply_contract():\n"
            "    assert multiply(3, 4) == 12\n"
        )

        rendered, meta = build_exact_test_context(
            test_code,
            failed_nodeids=["test_buggy.py::test_multiply_contract"],
            failure_output="",
            max_chars=140,
        )

        assert "def test_multiply_contract" in rendered
        assert "def test_addition_regression" not in rendered
        assert meta["source"] == "nodeids"
        assert meta["selected_nodeids"] == ["test_buggy.py::test_multiply_contract"]

    def test_selects_unittest_method_and_setup_methods(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        test_code = (
            "import unittest\n"
            "from buggy import multiply\n\n\n"
            "class TestMultiply(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        self.base = 3\n\n"
            "    def test_addition_regression(self):\n"
            "        self.assertEqual(multiply(self.base, 4), 7)\n\n"
            "    def test_multiply_contract(self):\n"
            "        self.assertEqual(multiply(self.base, 4), 12)\n"
        )

        rendered, _meta = build_exact_test_context(
            test_code,
            failed_nodeids=["test_buggy.py::TestMultiply::test_multiply_contract"],
            failure_output="",
            max_chars=240,
        )

        assert "class TestMultiply:" in rendered
        assert "def setUp" in rendered
        assert "def test_multiply_contract" in rendered
        assert "def test_addition_regression" not in rendered

    def test_merges_multiple_failed_methods_from_same_unittest_class(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        test_code = (
            "import unittest\n"
            "from buggy import multiply\n\n\n"
            "class TestMultiply(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        self.base = 3\n\n"
            "    def test_addition_regression(self):\n"
            "        self.assertEqual(multiply(self.base, 4), 7)\n\n"
            "    def test_multiply_contract(self):\n"
            "        self.assertEqual(multiply(self.base, 4), 12)\n"
        )

        rendered, _meta = build_exact_test_context(
            test_code,
            failed_nodeids=[
                "test_buggy.py::TestMultiply::test_addition_regression",
                "test_buggy.py::TestMultiply::test_multiply_contract",
            ],
            failure_output="",
            max_chars=260,
        )

        assert "def setUp" in rendered
        assert "def test_addition_regression" in rendered
        assert "def test_multiply_contract" in rendered

    def test_includes_referenced_fixture_helpers_and_classes_in_priority_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        test_code = (
            "import pytest\n"
            "from buggy import multiply\n\n\n"
            "@pytest.fixture\n"
            "def sample_value():\n"
            "    return 3\n\n\n"
            "def helper_expected(value):\n"
            "    return value * 4\n\n\n"
            "class Support:\n"
            "    @staticmethod\n"
            "    def compute(value):\n"
            "        return helper_expected(value)\n\n\n"
            "def test_multiply_contract(sample_value):\n"
            "    assert multiply(sample_value, 4) == helper_expected(sample_value)\n"
            "    assert multiply(sample_value, 4) == Support.compute(sample_value)\n"
        )

        rendered, meta = build_exact_test_context(
            test_code,
            failed_nodeids=["test_buggy.py::test_multiply_contract"],
            failure_output="",
            max_chars=420,
        )

        assert "def sample_value" in rendered
        assert "def helper_expected" in rendered
        assert "class Support" in rendered
        assert rendered.index("def sample_value") < rendered.index("def helper_expected")
        assert meta["helper_names"] == ["sample_value", "helper_expected", "Support"]

    def test_falls_back_to_line_anchor_slices_when_nodeids_are_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        test_code = "\n".join(f"line_{idx} = {idx}" for idx in range(1, 80))

        rendered, meta = build_exact_test_context(
            test_code,
            failed_nodeids=[],
            failure_output="AssertionError at test_buggy.py:40",
            max_chars=400,
        )

        assert "# test_buggy.py:" in rendered
        assert "line_40 = 40" in rendered
        assert meta["source"] == "line_anchors_fallback"

    def test_never_exceeds_requested_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_exact_test_context = _import_tracecoder_repo_module(
            monkeypatch, "src.exact_test_context"
        ).build_exact_test_context
        repeated_asserts = "\n".join(
            f"    assert multiply({idx}, {idx + 1}) == {idx * (idx + 1)}" for idx in range(100)
        )
        test_code = f"from buggy import multiply\n\n\ndef test_many_values():\n{repeated_asserts}\n"

        rendered, _meta = build_exact_test_context(
            test_code,
            failed_nodeids=["test_buggy.py::test_many_values"],
            failure_output="",
            max_chars=120,
        )

        assert len(rendered) <= 120


class TestTraceCoderLlmdEbugNativeWiring:
    def test_run_llmdebug_pytest_case_reports_failed_nodeids(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trace_runner = _import_tracecoder_repo_module(monkeypatch, "src.traceRunner")
        problem_data = {
            "files": [
                {
                    "path": "buggy.py",
                    "encoding": "utf-8",
                    "content": "def multiply(a, b):\n    return a + b\n",
                },
                {
                    "path": "test_buggy.py",
                    "encoding": "utf-8",
                    "content": (
                        "from buggy import multiply\n\n\n"
                        "def test_multiply_contract():\n"
                        "    assert multiply(3, 4) == 12\n"
                    ),
                },
            ],
            "python_args": ["-m", "pytest", "-q", "test_buggy.py"],
        }

        result = trace_runner.run_llmdebug_pytest_case(
            problem_data,
            "def multiply(a, b):\n    return a + b\n",
            timeout_seconds=10,
        )

        assert result["exitcode"] != 0
        assert result["failed_nodeids"] == ["test_buggy.py::test_multiply_contract"]

    def test_llmdebug_evaluation_preserves_failed_nodeids(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluation = _import_tracecoder_repo_module(
            monkeypatch,
            "datasets.llmdebug.evaluation",
            stub_pandas=True,
        )

        monkeypatch.setattr(
            evaluation,
            "run_llmdebug_pytest_case",
            lambda *_args, **_kwargs: {
                "exitcode": 1,
                "passed_count": 0,
                "failed_count": 1,
                "total_count": 1,
                "failed_nodeids": ["test_buggy.py::test_multiply_contract"],
                "stdout": "",
                "stderr": "AssertionError",
                "timed_out": False,
            },
        )

        result = evaluation.check_correctness(
            {"task_id": "case", "test": "placeholder"},
            "def multiply(a, b):\n    return a + b\n",
        )

        assert result["_debug_failed_nodeids"] == ["test_buggy.py::test_multiply_contract"]

    def test_analysis_prompt_includes_exact_test_context_only_in_analysis_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem_processor = _import_tracecoder_repo_module(
            monkeypatch,
            "problem_processor",
            stub_openai=True,
            stub_pandas=True,
        )
        problem_data = {"complete_prompt": "Repair buggy.py"}
        exact_test_context = "def test_multiply_contract():\n    assert multiply(3, 4) == 12"

        analysis_prompt = problem_processor._build_analysis_planning_prompt(
            problem_data,
            "def multiply(a, b):\n    return a + b\n",
            "AssertionError: values differ",
            [],
            exact_test_context=exact_test_context,
        )
        implementation_prompt = problem_processor._build_code_implementation_prompt(
            problem_data,
            "def multiply(a, b):\n    return a + b\n",
            "AssertionError: values differ",
            "Replace + with *.",
        )

        assert "Exact failing test context" in analysis_prompt
        assert exact_test_context in analysis_prompt
        assert "Exact failing test context" not in implementation_prompt
        assert exact_test_context not in implementation_prompt


class TestTraceCoderRepoPatchedModules:
    def test_failure_feedback_compaction_preserves_summary_and_tail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tracecoder_reporting = _import_tracecoder_repo_module(
            monkeypatch,
            "reporting",
            stub_pandas=True,
        )
        feedback = tracecoder_reporting.build_llmdebug_failure_feedback(
            passed_count=1,
            failed_count=1,
            total_count=2,
            exitcode=1,
            timed_out=False,
            stdout_text="stdout\n" + ("A" * 8000),
            stderr_text="stderr\n" + ("B" * 8000),
            error_text="AssertionError: values differ",
            max_chars=1200,
            tail_max_chars=300,
        )

        assert "Tests passed: 1/2" in feedback
        assert "Stdout tail:" in feedback
        assert "Stderr tail:" in feedback
        assert len(feedback) <= 1200

    def test_generator_sends_stage_specific_max_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tracecoder_generation = _import_tracecoder_repo_module(
            monkeypatch,
            "src.generation",
            stub_openai=True,
        )
        observed: dict[str, object] = {}

        class _Usage:
            prompt_tokens = 11
            completion_tokens = 7
            total_tokens = 18

        class _Response:
            def __init__(self) -> None:
                self.choices = [type("Choice", (), {"finish_reason": "stop"})()]
                self.usage = _Usage()

        class _ChatCompletions:
            def create(self, **kwargs: object) -> _Response:
                observed.update(kwargs)
                return _Response()

        fake_client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": _ChatCompletions()})()},
        )()

        monkeypatch.setattr(tracecoder_generation, "_build_client", lambda: fake_client)
        monkeypatch.setattr(
            tracecoder_generation,
            "time",
            type("T", (), {"sleep": staticmethod(lambda *_args, **_kwargs: None)})(),
        )
        monkeypatch.setattr(
            tracecoder_generation,
            "normalize_chat_completion_response",
            lambda response, **_kwargs: type(
                "Normalized",
                (),
                {
                    "content_text": "patched",
                    "reasoning_content": "",
                    "reasoning": "",
                },
            )(),
        )
        monkeypatch.setattr(
            tracecoder_generation,
            "require_final_content",
            lambda normalized, **_kwargs: normalized.content_text,
        )
        monkeypatch.setenv("TRACECODER_ANALYSIS_MAX_TOKENS", "2048")
        monkeypatch.setenv("TRACECODER_IMPLEMENTATION_MAX_TOKENS", "4096")

        result = tracecoder_generation.generator(
            "fix the code",
            "code4_generate",
            "test-model",
            progress_phase="analysis_planning",
        )

        assert result[0] == "patched"
        assert observed["max_tokens"] == 2048
