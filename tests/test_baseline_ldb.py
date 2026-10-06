"""Tests for the LDB baseline adapter and strict case converter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from baselines.adapter_base import (
    BaselineCaseContext,
    BaselineCaseIncompatibleError,
    BaselineConfig,
    BaselineResult,
)
from baselines.ldb.adapter import LDBAdapter
from baselines.ldb.case_converter import (
    _build_prompt,
    _extract_assertions,
    _extract_entry_point,
    convert_case,
)
from baselines.ldb.result_extractor import make_diff
from baselines.subprocess_monitor import MonitoredSubprocessResult


@pytest.fixture()
def simple_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "my_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a - b\n',
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return case_dir


@pytest.fixture()
def class_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "class_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a - b\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import Calculator\n\n\ndef test_add():\n"
        "    calc = Calculator()\n    assert calc.add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return case_dir


@pytest.fixture()
def subprocess_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "subprocess_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "import subprocess\n\n\ndef test_cli():\n"
        "    subprocess.run(['python', 'buggy.py'], check=False)\n",
        encoding="utf-8",
    )
    return case_dir


def _make_ldb_repo(tmp_path: Path, *, with_dataset_support: bool = True) -> Path:
    repo = tmp_path / "fake_ldb"
    programming = repo / "programming"
    programming.mkdir(parents=True)
    dataset_marker = '"llmdebug"' if with_dataset_support else '"HumanEval"'
    (programming / "main.py").write_text("# stub\n", encoding="utf-8")
    (programming / "ldb.py").write_text(
        (
            "DATASET = "
            f"{dataset_marker}\n"
            "llm_duration_sec = 0.0\n"
            "usage_available = True\n"
            "is_passing = False\n"
            "test_feedback = []\n"
            if with_dataset_support
            else f"DATASET = {dataset_marker}\n"
        ),
        encoding="utf-8",
    )
    (programming / "utils.py").write_text(f"DATASET = {dataset_marker}\n", encoding="utf-8")
    generators = programming / "generators"
    generators.mkdir()
    (generators / "model.py").write_text(
        (
            "_record_metrics = True\n"
            "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE\n"
            "LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL\n"
            "compat_error\n"
            "super().__init__(model_name)\n"
        )
        if with_dataset_support
        else "legacy = True\n",
        encoding="utf-8",
    )
    (generators / "py_generate.py").write_text(
        (
            'dataset_type in ["HumanEval", "MBPP", "llmdebug"]\n'
            if with_dataset_support
            else 'dataset_type in ["HumanEval", "MBPP"]\n'
        ),
        encoding="utf-8",
    )
    return repo


def _patch_monitored_run(monkeypatch: pytest.MonkeyPatch, fake_run) -> None:
    def _fake_monitored_run(
        *,
        cmd: list[str],
        cwd: Path | str | None,
        env: dict[str, str] | None,
        timeout_sec: int,
        **_kwargs,
    ) -> MonitoredSubprocessResult:
        try:
            completed = fake_run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
                cwd=str(cwd),
                env=dict(env or {}),
            )
        except subprocess.TimeoutExpired as exc:
            return MonitoredSubprocessResult(
                returncode=-9,
                duration_sec=float(timeout_sec),
                stdout=str(exc.output or ""),
                stderr=str(exc.stderr or ""),
                timed_out=True,
                timeout_sec=int(timeout_sec),
            )
        return MonitoredSubprocessResult(
            returncode=completed.returncode,
            duration_sec=0.01,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr("baselines.ldb.adapter.run_monitored_subprocess", _fake_monitored_run)


class TestCaseConverter:
    def test_convert_case_basic(self, simple_case: Path) -> None:
        result = convert_case(simple_case)

        assert result.task_id == "llmdebug/my_case"
        assert result.entry_point == "add"
        assert result.dataset_type == "llmdebug"
        assert result.seed == (simple_case / "buggy.py").read_text(encoding="utf-8")
        assert result.case_information == {
            "case_info_policy": "neutral_v1",
            "test_source_exposure": "asserts",
            "prompt_files": ["buggy.py", "test_buggy.py"],
        }
        assert result.test.startswith("def check(candidate):\n")
        assert result.given_tests == ["assert add(1, 2) == 3\n"]

    def test_convert_case_prefers_case_context_case_id(self, simple_case: Path) -> None:
        adapter = LDBAdapter()

        result = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="canonical_case_id",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=9,
            ),
        )

        assert result.task_id == "llmdebug/canonical_case_id"

    def test_convert_case_rejects_non_function_entry_points(self, class_case: Path) -> None:
        with pytest.raises(
            BaselineCaseIncompatibleError, match="non_function_entry_point_unsupported"
        ):
            convert_case(class_case)

    def test_convert_case_rejects_subprocess_cases(self, subprocess_case: Path) -> None:
        with pytest.raises(BaselineCaseIncompatibleError, match="subprocess_task_unsupported"):
            convert_case(subprocess_case)

    def test_extract_entry_point_function(self) -> None:
        assert _extract_entry_point("def solve(x):\n    return x\n", "") == "solve"

    def test_extract_entry_point_import_fallback(self) -> None:
        assert _extract_entry_point("x = 1\n", "from buggy import compute\n") == "compute"

    def test_build_prompt_with_docstring(self) -> None:
        code = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
        prompt = _build_prompt(code, "add")
        assert "def add(a, b):" in prompt
        assert '"""Add two numbers."""' in prompt

    def test_build_prompt_without_definition_is_bare_signature(self) -> None:
        assert _build_prompt("x = 1\n", "compute") == "def compute():"

    def test_build_prompt_syntax_error_is_bare_signature(self) -> None:
        assert _build_prompt("def broken(:\n", "compute") == "def compute():"

    def test_convert_case_without_docstring_adds_no_description(  # ATT-02 / D1
        self, tmp_path: Path
    ) -> None:
        case_dir = tmp_path / "no_docstring_case"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text("def compute(x):\n    return x + 1\n", encoding="utf-8")
        (case_dir / "test_buggy.py").write_text(
            "from buggy import compute\n\n\ndef test_compute():\n    assert compute(2) == 2\n",
            encoding="utf-8",
        )

        result = convert_case(case_dir)

        assert result.prompt == "def compute(x):"

    def test_extract_assertions_returns_executable_snippet(self) -> None:
        test_code = "from buggy import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        snippets = _extract_assertions(test_code, "def add(a, b):\n    return a - b\n")
        assert snippets == ["assert add(1, 2) == 3\n"]

    def test_convert_case_rejects_top_level_test_support_code(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "support_code_case"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (case_dir / "test_buggy.py").write_text(
            "import math\nfrom buggy import add\n\n\ndef test_add():\n"
            "    assert add(1, 2) == math.ceil(2.1)\n",
            encoding="utf-8",
        )

        with pytest.raises(BaselineCaseIncompatibleError, match="test_support_code_unsupported"):
            convert_case(case_dir)

    def test_convert_case_rejects_top_level_helper_support_code(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "helper_support_case"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (case_dir / "test_buggy.py").write_text(
            "from buggy import add\n\n\ndef helper():\n    return add(1, 2)\n\n\ndef test_add():\n"
            "    assert helper() == 3\n",
            encoding="utf-8",
        )

        with pytest.raises(BaselineCaseIncompatibleError, match="test_support_code_unsupported"):
            convert_case(case_dir)

    def test_convert_case_rejects_non_assert_test_bodies(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "non_assert_body_case"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (case_dir / "test_buggy.py").write_text(
            "from buggy import add\n\n\ndef test_add():\n"
            "    result = add(1, 2)\n"
            "    assert result == 3\n",
            encoding="utf-8",
        )

        with pytest.raises(BaselineCaseIncompatibleError, match="non_assert_test_body_unsupported"):
            convert_case(case_dir)


class TestResultExtractor:
    def test_make_diff_produces_valid_diff(self, simple_case: Path) -> None:
        repaired = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
        diff = make_diff(simple_case / "buggy.py", repaired)

        assert diff is not None
        assert "diff --git" in diff
        assert "-    return a - b" in diff
        assert "+    return a + b" in diff

    def test_make_diff_identical_returns_none(self, simple_case: Path) -> None:
        same_code = (simple_case / "buggy.py").read_text(encoding="utf-8")
        assert make_diff(simple_case / "buggy.py", same_code) is None


class TestLDBAdapter:
    def test_adapter_name(self) -> None:
        assert LDBAdapter().name == "ldb"

    def test_setup_raises_on_missing_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            LDBAdapter().setup(BaselineConfig(repo_dir=tmp_path / "missing"))

    def test_setup_requires_llmdebug_dataset_patch(self, tmp_path: Path) -> None:
        repo = _make_ldb_repo(tmp_path, with_dataset_support=False)
        with pytest.raises(RuntimeError, match="llmdebug compatibility support"):
            LDBAdapter().setup(BaselineConfig(repo_dir=repo))

    def test_setup_succeeds_with_valid_repo(self, tmp_path: Path) -> None:
        repo = _make_ldb_repo(tmp_path)
        adapter = LDBAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))
        assert adapter.repo_commit == ""

    def test_convert_case_delegates(self, simple_case: Path) -> None:
        result = LDBAdapter().convert_case(simple_case)
        assert result.task_id == "llmdebug/my_case"
        assert result.entry_point == "add"

    def test_run_repair_uses_upstream_cli_contract(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_ldb_repo(tmp_path)
        adapter = LDBAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            api_base_url="http://127.0.0.1:8080",
            max_attempts=3,
            timeout_sec=9,
        )
        adapter._config = config
        observed: dict[str, object] = {}
        observed_rows: dict[str, dict[str, object]] = {}

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            observed["cmd"] = cmd
            observed["cwd"] = cwd
            observed["timeout"] = timeout
            observed["env"] = env

            run_name = cmd[cmd.index("--run_name") + 1]
            root_dir = Path(cmd[cmd.index("--root_dir") + 1])
            dataset_path = Path(cmd[cmd.index("--dataset_path") + 1])
            seed_path = Path(cmd[cmd.index("--seedfile") + 1])
            test_path = Path(cmd[cmd.index("--testfile") + 1])
            observed_rows["dataset"] = json.loads(
                dataset_path.read_text(encoding="utf-8").splitlines()[0]
            )
            observed_rows["seed"] = json.loads(
                seed_path.read_text(encoding="utf-8").splitlines()[0]
            )
            observed_rows["test"] = json.loads(
                test_path.read_text(encoding="utf-8").splitlines()[0]
            )
            log_dir = root_dir / run_name
            log_dir.mkdir(parents=True)
            (log_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "llmdebug/my_case",
                        "solution": "def add(a, b):\n    return a + b\n",
                        "is_passing": True,
                        "is_solved": True,
                        "debug_iter": 2,
                        "test_feedback": [["test_add"], []],
                        "llm_call_count": 4,
                        "llm_duration_sec": 1.75,
                        "eval_duration_sec": 0.8,
                        "processing_time_seconds": 3.0,
                        "usage_available": True,
                        "tokens_prompt": 31,
                        "tokens_completion": 14,
                        "tokens_total": 45,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        case_input = convert_case(simple_case).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:2 in add",
                    "top_facts": ["Crash is localized to function add"],
                }
            },
            "history_bundle": {
                "lessons": [
                    {"source": "attempt_history", "attempt": 1, "lesson": "preserve add contract"}
                ]
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["touch the crash file"],
            },
            "solver_hints": {
                "retry_plan": "Focus on the add return expression.",
            },
        }

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.repaired_code == "def add(a, b):\n    return a + b\n"
        assert result.iterations_used == 2
        assert result.token_usage == {
            "prompt_tokens": 31,
            "completion_tokens": 14,
            "total_tokens": 45,
        }
        metrics = result.meta["metrics"]
        assert metrics["llm_call_count"] == 4
        assert metrics["llm_duration_sec"] == 1.75
        assert metrics["eval_duration_sec"] == 0.8
        assert metrics["internal_total_duration_sec"] == 3.0
        assert metrics["orchestration_duration_sec"] == pytest.approx(0.45)
        assert metrics["usage_available"] is True
        assert metrics["is_passing"] is True
        assert metrics["test_feedback_steps"] == 2
        assert metrics["test_feedback_nonempty_steps"] == 1
        assert metrics["test_feedback_improvement_steps"] == 1
        assert metrics["test_feedback_regression_steps"] == 0
        assert metrics["test_feedback_stagnation_steps"] == 0
        assert metrics["test_feedback_final_fail_count"] == 0
        assert metrics["usage_source"] == "provider_reported"

        cmd = observed["cmd"]
        assert isinstance(cmd, list)
        assert "--dataset_path" in cmd
        assert "--run_name" in cmd
        assert "--root_dir" in cmd
        assert "--seedfile" in cmd
        assert "--testfile" in cmd
        assert "--pass_at_k" in cmd
        assert "--dataset" not in cmd
        assert "--output_dir" not in cmd
        assert observed["cwd"] == str(repo / "programming")
        env = observed["env"]
        assert isinstance(env, dict)
        assert env["LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE"] == "auto"
        assert env["LLMDEBUG_BASELINE_OPENAI_ALLOW_REASONING_FALLBACK"] == "0"
        assert env["LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL"] == "test-model"
        assert str(Path(__file__).resolve().parents[1]) in env["PYTHONPATH"]
        assert cmd[cmd.index("--model") + 1] == "gpt-4"

        dataset_row = observed_rows["dataset"]
        seed_row = observed_rows["seed"]
        test_row = observed_rows["test"]
        assert dataset_row["task_id"] == "llmdebug/my_case"
        assert "LLMDebug controller guidance:" in dataset_row["prompt"]
        assert "crash_anchor: buggy.py:2 in add" in dataset_row["prompt"]
        assert dataset_row["test"].startswith("def check(candidate):")
        assert seed_row["solution"].startswith("def add(a, b):")
        assert test_row["given_tests"] == ["assert add(1, 2) == 3\n"]

    def test_extract_diff_with_repair(self, simple_case: Path) -> None:
        result = BaselineResult(
            repaired_code='def add(a, b):\n    """Add two numbers."""\n    return a + b\n',
            success=True,
            iterations_used=1,
        )
        diff = LDBAdapter().extract_diff(simple_case, result)
        assert diff is not None
        assert "+    return a + b" in diff

    def test_run_repair_uses_fair_v1_hybrid_guidance_block(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_ldb_repo(tmp_path)
        adapter = LDBAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=1,
            extra={"baseline_hybrid_guidance_mode": "fair_v1"},
        )
        adapter._config = config
        observed_rows: dict[str, dict[str, object]] = {}

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            dataset_path = Path(cmd[cmd.index("--dataset_path") + 1])
            seed_path = Path(cmd[cmd.index("--seedfile") + 1])
            test_path = Path(cmd[cmd.index("--testfile") + 1])
            root_dir = Path(cmd[cmd.index("--root_dir") + 1])
            run_name = cmd[cmd.index("--run_name") + 1]
            observed_rows["dataset"] = json.loads(
                dataset_path.read_text(encoding="utf-8").splitlines()[0]
            )
            observed_rows["seed"] = json.loads(
                seed_path.read_text(encoding="utf-8").splitlines()[0]
            )
            observed_rows["test"] = json.loads(
                test_path.read_text(encoding="utf-8").splitlines()[0]
            )
            log_dir = root_dir / run_name
            log_dir.mkdir(parents=True)
            (log_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "llmdebug/my_case",
                        "solution": "def add(a, b):\n    return a + b\n",
                        "is_passing": True,
                        "is_solved": True,
                        "debug_iter": 1,
                        "test_feedback": [["test_add"], []],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        case_input = convert_case(simple_case).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:2 in add",
                    "top_facts": ["Focus on buggy.py", "Do not widen the edit"],
                }
            },
            "history_bundle": {
                "lessons": [
                    {"source": "attempt_history", "attempt": 1, "lesson": "preserve add contract"}
                ]
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["touch the crash file", "preserve the function signature"],
            },
            "solver_hints": {
                "retry_plan": "Edit only the add return expression.",
                "strategy_summary": "should not appear",
                "decomposition": [{"title": "localize", "question": "should not appear"}],
            },
        }

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        dataset_prompt = str(observed_rows["dataset"]["prompt"])
        assert "LLMDebug controller guidance:" in dataset_prompt
        assert "verifier: status=previous_verifier_rejected" in dataset_prompt
        assert "retry_plan: Edit only the add return expression." in dataset_prompt
        assert "strategy_summary:" not in dataset_prompt
        assert "decomposition:" not in dataset_prompt

    def test_parse_output_no_jsonl(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "empty"
        log_dir.mkdir()
        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)
        assert result.repaired_code is None
        assert result.meta.get("error") == "no_output_jsonl"

    def test_parse_output_no_jsonl_uses_sidecar_metrics(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "empty"
        log_dir.mkdir()
        (log_dir / "llmdebug_metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "baseline": "ldb",
                    "llm_call_count": 2,
                    "llm_duration_sec": 0.9,
                    "eval_duration_sec": 0.6,
                    "internal_total_duration_sec": 2.0,
                    "orchestration_duration_sec": 0.5,
                    "usage_available": True,
                    "usage_reported_call_count": 2,
                    "usage_missing_call_count": 0,
                    "tokens_prompt": 17,
                    "tokens_completion": 8,
                    "tokens_total": 25,
                    "reported_tokens_prompt": 17,
                    "reported_tokens_completion": 8,
                    "reported_tokens_total": 25,
                    "prompt_chars_total": 112,
                    "completion_chars_total": 41,
                    "chars_total": 153,
                    "reported_chars_prompt": 112,
                    "reported_chars_completion": 41,
                    "reported_chars_total": 153,
                    "attempts_completed": 2,
                    "usage_source": "provider_reported",
                    "is_passing": True,
                }
            ),
            encoding="utf-8",
        )

        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)

        assert result.repaired_code is None
        assert result.success is False
        assert result.iterations_used == 2
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 8,
            "total_tokens": 25,
        }
        assert result.meta["error"] == "no_output_jsonl"
        assert result.meta["metrics"] == {
            "llm_call_count": 2,
            "llm_duration_sec": 0.9,
            "eval_duration_sec": 0.6,
            "internal_total_duration_sec": 2.0,
            "orchestration_duration_sec": 0.5,
            "usage_available": True,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 0,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 8,
            "reported_tokens_total": 25,
            "prompt_chars_total": 112,
            "completion_chars_total": 41,
            "chars_input": 112,
            "chars_output": 41,
            "chars_total": 153,
            "reported_chars_prompt": 112,
            "reported_chars_completion": 41,
            "reported_chars_total": 153,
            "usage_source": "provider_reported",
            "is_passing": True,
        }

    def test_parse_output_no_jsonl_surfaces_api_error_from_stderr(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "empty"
        log_dir.mkdir()

        result = LDBAdapter()._parse_output(
            log_dir=log_dir,
            stdout="",
            stderr="openai.APIConnectionError: Connection error.\n",
            returncode=1,
        )

        assert result.repaired_code is None
        assert result.success is False
        assert result.meta["error"] == "api_error"
        assert result.meta["root_cause_error"] == "api_error"
        assert result.meta["artifact_error"] == "no_output_jsonl"

    def test_parse_output_no_jsonl_zero_call_sidecar_does_not_expose_usage(
        self, tmp_path: Path
    ) -> None:
        log_dir = tmp_path / "empty"
        log_dir.mkdir()
        (log_dir / "llmdebug_metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "baseline": "ldb",
                    "llm_call_count": 0,
                    "llm_duration_sec": 0.0,
                    "eval_duration_sec": 0.6,
                    "internal_total_duration_sec": 1.0,
                    "orchestration_duration_sec": 0.4,
                    "usage_available": True,
                    "usage_reported_call_count": 0,
                    "usage_missing_call_count": 0,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "tokens_total": 0,
                    "reported_tokens_prompt": 0,
                    "reported_tokens_completion": 0,
                    "reported_tokens_total": 0,
                    "attempts_completed": 0,
                    "is_passing": False,
                }
            ),
            encoding="utf-8",
        )

        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)

        assert result.repaired_code is None
        assert result.success is False
        assert result.iterations_used == 0
        assert result.token_usage == {}
        assert result.meta["error"] == "no_output_jsonl"
        assert result.meta["metrics"]["llm_call_count"] == 0
        assert result.meta["metrics"]["usage_available"] is False

    def test_parse_output_valid_jsonl(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "output"
        log_dir.mkdir()
        (log_dir / "results.jsonl").write_text(
            json.dumps(
                {
                    "task_id": "llmdebug/my_case",
                    "solution": "def add(a, b):\n    return a + b\n",
                    "is_passing": True,
                    "is_solved": True,
                    "debug_iter": 2,
                    "test_feedback": [["test_add"], []],
                    "llm_call_count": 2,
                    "llm_duration_sec": 0.9,
                    "eval_duration_sec": 0.6,
                    "processing_time_seconds": 2.0,
                    "usage_available": True,
                    "tokens_prompt": 17,
                    "tokens_completion": 8,
                    "tokens_total": 25,
                    "prompt_chars_total": 112,
                    "completion_chars_total": 41,
                    "chars_total": 153,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)
        assert result.repaired_code == "def add(a, b):\n    return a + b\n"
        assert result.success is True
        assert result.iterations_used == 2
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 8,
            "total_tokens": 25,
        }
        assert result.meta["metrics"]["llm_call_count"] == 2
        assert result.meta["metrics"]["prompt_chars_total"] == 112
        assert result.meta["metrics"]["completion_chars_total"] == 41
        assert result.meta["metrics"]["chars_total"] == 153
        assert result.meta["metrics"]["reported_chars_prompt"] == 112
        assert result.meta["metrics"]["reported_chars_completion"] == 41
        assert result.meta["metrics"]["reported_chars_total"] == 153
        assert result.meta["metrics"]["is_passing"] is True
        assert result.meta["metrics"]["test_feedback_steps"] == 2

    def test_parse_output_surfaces_openai_compat_reasoning_only_error(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "output"
        log_dir.mkdir()
        (log_dir / "results.jsonl").write_text(
            json.dumps(
                {
                    "task_id": "llmdebug/my_case",
                    "solution": "def add(a, b):\n    return a - b\n",
                    "is_passing": False,
                    "is_solved": False,
                    "debug_iter": 0,
                    "test_feedback": [["test_add"]],
                    "error": "openai_compat_reasoning_only",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (log_dir / "llmdebug_metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "baseline": "ldb",
                    "compat_message": "Disable thinking in the server-side template.",
                    "compat_profile_applied": "lmstudio_model_yaml",
                    "empty_content_with_reasoning": True,
                    "reasoning_present": True,
                }
            ),
            encoding="utf-8",
        )

        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)

        assert result.success is False
        assert result.meta["error"] == "openai_compat_reasoning_only"
        assert result.meta["compat_message"] == "Disable thinking in the server-side template."
        assert result.meta["compat_meta"]["compat_profile_applied"] == "lmstudio_model_yaml"

    def test_parse_output_rejects_missing_required_diagnostics(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "output"
        log_dir.mkdir()
        (log_dir / "results.jsonl").write_text(
            json.dumps(
                {
                    "task_id": "llmdebug/my_case",
                    "solution": "def add(a, b):\n    return a + b\n",
                    "is_solved": True,
                    "debug_iter": 2,
                    "llm_call_count": 2,
                    "llm_duration_sec": 0.9,
                    "eval_duration_sec": 0.6,
                    "processing_time_seconds": 2.0,
                    "usage_available": True,
                    "tokens_prompt": 17,
                    "tokens_completion": 8,
                    "tokens_total": 25,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)

        assert result.success is False
        assert result.repaired_code is None
        assert result.meta["error"] == "baseline_incompatible_output_schema"
        assert result.meta["schema_missing_fields"] == ["is_passing", "test_feedback"]

    def test_parse_output_empty_solution(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "output"
        log_dir.mkdir()
        (log_dir / "results.jsonl").write_text(
            json.dumps({"task_id": "llmdebug/my_case", "solution": "", "completion": ""}) + "\n",
            encoding="utf-8",
        )

        result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)
        assert result.repaired_code is None
        assert result.meta.get("error") == "baseline_incompatible_output_schema"

    def test_run_repair_timeout_recovers_metrics_from_sidecar(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_ldb_repo(tmp_path)
        adapter = LDBAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            api_base_url="http://127.0.0.1:8080",
            max_attempts=3,
            timeout_sec=9,
        )
        adapter._config = config

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            run_name = cmd[cmd.index("--run_name") + 1]
            root_dir = Path(cmd[cmd.index("--root_dir") + 1])
            log_dir = root_dir / run_name
            log_dir.mkdir(parents=True)
            (log_dir / "llmdebug_metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "ldb",
                        "llm_call_count": 4,
                        "llm_duration_sec": 1.75,
                        "eval_duration_sec": 0.8,
                        "internal_total_duration_sec": 3.0,
                        "orchestration_duration_sec": 0.45,
                        "usage_available": True,
                        "usage_reported_call_count": 4,
                        "usage_missing_call_count": 0,
                        "tokens_prompt": 31,
                        "tokens_completion": 14,
                        "tokens_total": 45,
                        "reported_tokens_prompt": 31,
                        "reported_tokens_completion": 14,
                        "reported_tokens_total": 45,
                        "prompt_chars_total": 205,
                        "completion_chars_total": 73,
                        "chars_total": 278,
                        "reported_chars_prompt": 205,
                        "reported_chars_completion": 73,
                        "reported_chars_total": 278,
                        "attempts_completed": 2,
                        "usage_source": "provider_reported",
                        "is_passing": True,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.iterations_used == 2
        assert result.token_usage == {
            "prompt_tokens": 31,
            "completion_tokens": 14,
            "total_tokens": 45,
        }
        assert result.meta["error"] == "timeout"
        assert result.meta["metrics"] == {
            "llm_call_count": 4,
            "llm_duration_sec": 1.75,
            "eval_duration_sec": 0.8,
            "internal_total_duration_sec": 3.0,
            "orchestration_duration_sec": 0.45,
            "usage_available": True,
            "usage_reported_call_count": 4,
            "usage_missing_call_count": 0,
            "reported_tokens_prompt": 31,
            "reported_tokens_completion": 14,
            "reported_tokens_total": 45,
            "prompt_chars_total": 205,
            "completion_chars_total": 73,
            "chars_input": 205,
            "chars_output": 73,
            "chars_total": 278,
            "reported_chars_prompt": 205,
            "reported_chars_completion": 73,
            "reported_chars_total": 278,
            "usage_source": "provider_reported",
            "is_passing": True,
        }

    def test_run_repair_emits_sidecar_progress_updates(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_ldb_repo(tmp_path)
        adapter = LDBAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            api_base_url="http://127.0.0.1:8080",
            max_attempts=3,
            timeout_sec=9,
        )
        adapter._config = config
        events: list[tuple[str, dict[str, object] | None]] = []
        adapter.set_progress_callback(lambda event, payload: events.append((event, payload)))

        def _fake_monitored_run(
            *,
            cmd: list[str],
            progress_callback,
            progress_event: str,
            state_supplier,
            **_kwargs,
        ) -> MonitoredSubprocessResult:
            run_name = cmd[cmd.index("--run_name") + 1]
            root_dir = Path(cmd[cmd.index("--root_dir") + 1])
            log_dir = root_dir / run_name
            log_dir.mkdir(parents=True)
            (log_dir / "llmdebug_metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "ldb",
                        "status": "running",
                        "llm_call_count": 2,
                        "eval_duration_sec": 0.5,
                        "attempts_completed": 1,
                        "is_passing": False,
                    }
                ),
                encoding="utf-8",
            )
            payload = state_supplier()
            assert payload is not None
            progress_callback(progress_event, payload)
            (log_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "llmdebug/my_case",
                        "solution": "def add(a, b):\n    return a + b\n",
                        "is_passing": True,
                        "is_solved": True,
                        "debug_iter": 1,
                        "test_feedback": [[]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return MonitoredSubprocessResult(
                returncode=0,
                duration_sec=0.05,
                stdout="",
                stderr="",
            )

        monkeypatch.setattr("baselines.ldb.adapter.run_monitored_subprocess", _fake_monitored_run)

        result = adapter.run_repair(convert_case(simple_case).to_dict(), config)

        assert result.success is True
        progress_payloads = [payload for event, payload in events if event == "baseline_progress"]
        assert progress_payloads
        assert progress_payloads[-1] == {
            "owner": "baseline",
            "source": "sidecar",
            "phase_label": "baseline: ldb subprocess",
            "waiting_for": "ldb_subprocess",
            "external_operation": "baseline_repair",
            "external_provider": "ldb",
            "attempts_completed": 1,
            "llm_calls_observed": 2,
            "eval_duration_sec_observed": 0.5,
            "note": "status=running is_passing=false",
        }

    def test_run_repair_returns_structured_error_for_missing_python_executable(
        self,
        simple_case: Path,
        tmp_path: Path,
    ) -> None:
        repo = _make_ldb_repo(tmp_path)
        adapter = LDBAdapter()
        missing_python = tmp_path / "missing-python"
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=missing_python,
            model="test-model",
            max_attempts=1,
            timeout_sec=5,
        )
        adapter.setup(config)

        result = adapter.run_repair(convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.meta["error"] == "ldb_not_found"
        assert str(missing_python) in str(result.meta["exc"])


def test_bas15_unrepaired_row_still_reports_metrics_and_usage(tmp_path: Path) -> None:
    log_dir = tmp_path / "output"
    log_dir.mkdir()
    (log_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "task_id": "llmdebug/my_case",
                "solution": "",
                "is_passing": False,
                "is_solved": False,
                "debug_iter": 2,
                "test_feedback": [["test_add"], ["test_add"]],
                "llm_call_count": 3,
                "llm_duration_sec": 1.5,
                "eval_duration_sec": 0.4,
                "processing_time_seconds": 2.5,
                "usage_available": True,
                "tokens_prompt": 40,
                "tokens_completion": 12,
                "tokens_total": 52,
                "prompt_chars_total": 300,
                "completion_chars_total": 90,
                "chars_total": 390,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = LDBAdapter()._parse_output(log_dir=log_dir, stdout="", stderr="", returncode=0)

    assert result.repaired_code is None
    assert result.meta["error"] == "no_solution_in_output"
    assert result.meta["metrics"]["llm_call_count"] == 3
    assert result.meta["metrics"]["prompt_chars_total"] == 300
    assert result.token_usage


def test_bas16_model_routing_matches_gpt_family_exactly() -> None:
    from baselines.ldb.adapter import _resolve_upstream_model_selector

    assert _resolve_upstream_model_selector("gpt-4") == ("gpt-4", "gpt-4")
    assert _resolve_upstream_model_selector("gpt-4o-mini") == ("gpt-4o-mini", "gpt-4o-mini")
    assert _resolve_upstream_model_selector("gpt-3.5-turbo-0613")[0] == "gpt-3.5-turbo-0613"
    # Names that merely contain "gpt-4" route through the GPT path with the real id.
    assert _resolve_upstream_model_selector("my-gpt-4-finetune") == ("gpt-4", "my-gpt-4-finetune")
    assert _resolve_upstream_model_selector("qwen3-coder") == ("gpt-4", "qwen3-coder")
    assert _resolve_upstream_model_selector("starcoder") == ("starcoder", "starcoder")
