"""Tests for the TraceCoder baseline adapter and patched-case converter."""

from __future__ import annotations

import csv
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
from baselines.tracecoder.adapter import TraceCoderAdapter
from baselines.tracecoder.case_converter import (
    _build_prompt,
    _extract_entry_point,
    compact_problem_prompt,
    convert_case,
)
from baselines.tracecoder.result_extractor import make_diff
from evals.subprocess_monitor import MonitoredSubprocessResult

_TRACECODER_METRICS_PATCH = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "tracecoder"
    / "patches"
    / "0001-add-llmdebug-repair-mode.patch"
)


@pytest.fixture()
def simple_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "my_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def multiply(a, b):\n    return a + b\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text("CONSTANT = 12\n", encoding="utf-8")
    return case_dir


def _make_tracecoder_repo(
    tmp_path: Path,
    *,
    with_llmdebug_support: bool = True,
    with_metrics_sidecar_support: bool = False,
) -> Path:
    repo = tmp_path / "fake_tracecoder"
    (repo / "src").mkdir(parents=True)
    (repo / "datasets" / "llmdebug").mkdir(parents=True)
    config_lines = [
        'DATASET_PATHS = {"humaneval": {"data_path": "x", "eval_module": "datasets.human_eval.execution"}}',
        "def setup_arg_parser():\n    pass\n",
    ]
    if with_llmdebug_support:
        config_lines = [
            'DATASET_PATHS = {"llmdebug": {"data_path": "x", "eval_module": "datasets.llmdebug.evaluation"}}',
            "# --dataset-path support\n",
        ]
        if with_metrics_sidecar_support:
            config_lines.append("# --llmdebug-metrics-path support\n")
    (repo / "config.py").write_text("".join(config_lines), encoding="utf-8")
    processor_text = "legacy\n"
    if with_llmdebug_support:
        processor_text = (
            "repair_seeded\n"
            "execution_mode\n"
            "method_instrumentation\n"
            "method_two_step_repair\n"
            "direct_gen_passed\n"
            "final_passed\n"
            "stopped_reason\n"
            "total_llm_duration_sec\n"
            "total_eval_duration_sec\n"
            "build_exact_test_context\n"
            "exact_test_context_meta\n"
        )
    (repo / "problem_processor.py").write_text(processor_text, encoding="utf-8")
    runner_text = "print('stub')\n"
    if with_metrics_sidecar_support:
        runner_text += "llmdebug_metrics_path\n"
    (repo / "trace_learn_coder.py").write_text(runner_text, encoding="utf-8")
    generation_text = "print('generation')\n"
    if with_llmdebug_support:
        generation_text += (
            "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE\n"
            "openai_compat_reasoning_only\n"
            "TRACECODER_ANALYSIS_MAX_TOKENS\n"
            "TRACECODER_IMPLEMENTATION_MAX_TOKENS\n"
        )
    (repo / "src" / "generation.py").write_text(generation_text, encoding="utf-8")
    (repo / "src" / "traceRunner.py").write_text("print('runner')\n", encoding="utf-8")
    (repo / "src" / "exact_test_context.py").write_text(
        "def build_exact_test_context(*args, **kwargs):\n    return '', {}\n",
        encoding="utf-8",
    )
    reporting_text = "print('reporting')\n"
    if with_llmdebug_support:
        reporting_text += "build_llmdebug_failure_feedback\n"
    if with_metrics_sidecar_support:
        reporting_text += "save_metrics_sidecar\n"
    (repo / "reporting.py").write_text(reporting_text, encoding="utf-8")
    if with_llmdebug_support:
        (repo / "datasets" / "llmdebug" / "evaluation.py").write_text(
            "def check_correctness(*args, **kwargs):\n    return {}\n", encoding="utf-8"
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

    monkeypatch.setattr(
        "baselines.tracecoder.adapter.run_monitored_subprocess",
        _fake_monitored_run,
    )


class TestCaseConverter:
    def test_convert_case_packages_workspace_and_context(self, simple_case: Path) -> None:
        context = BaselineCaseContext(
            case_id="my_case",
            python_args=("-m", "pytest", "-q"),
            timeout_sec=11,
            description="Multiply two numbers.",
            category="logic_error",
            difficulty="easy",
            snapshot_dependency="none",
        )

        result = convert_case(simple_case, case_context=context)

        assert result.task_id == "my_case"
        assert result.entry_point == "multiply"
        assert result.repair_seeded is True
        assert result.execution_mode == "llmdebug_native"
        assert result.python_args == ["-m", "pytest", "-q"]
        assert result.timeout_sec == 11
        assert result.buggy_code == (simple_case / "buggy.py").read_text(encoding="utf-8")
        assert "Reference test summary:" in str(result.complete_prompt)
        paths = {row["path"] for row in result.files}
        assert paths == {"buggy.py", "helper.py", "test_buggy.py"}

    def test_convert_case_prefers_case_context_case_id(self, simple_case: Path) -> None:
        adapter = TraceCoderAdapter()

        result = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="canonical_case_id",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=11,
                description="Multiply two numbers.",
            ),
        )

        assert result.task_id == "canonical_case_id"

    def test_convert_case_rejects_non_pytest_commands(self, simple_case: Path) -> None:
        context = BaselineCaseContext(
            case_id="my_case",
            python_args=("buggy.py",),
            timeout_sec=10,
        )
        with pytest.raises(BaselineCaseIncompatibleError, match="non_pytest_command_unsupported"):
            convert_case(simple_case, case_context=context)

    def test_extract_entry_point_prefers_buggy_definition(self) -> None:
        assert _extract_entry_point("def compute(x):\n    return x\n", "") == "compute"

    def test_extract_entry_point_falls_back_to_import(self) -> None:
        assert _extract_entry_point("x = 1\n", "from buggy import transform\n") == "transform"

    def test_build_prompt_includes_description_and_tests(self) -> None:
        prompt = _build_prompt(
            buggy_code="def multiply(a, b):\n    return a + b\n",
            test_code="from buggy import multiply\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
            description="Multiply two numbers.",
            entry_point="multiply",
        )
        assert "Repair the provided buggy.py implementation" in prompt
        assert "Description: Multiply two numbers." in prompt
        assert "Reference test summary:" in prompt
        assert "multiply(3, 4) == 12" in prompt

    def test_build_prompt_compacts_large_test_buggy_inputs(self) -> None:
        repeated_asserts = "\n".join(
            f"    assert multiply({idx}, {idx + 1}) == {idx * (idx + 1)}" for idx in range(2000)
        )
        huge_test = f"from buggy import multiply\n\ndef test_many_cases():\n{repeated_asserts}\n"

        prompt = _build_prompt(
            buggy_code="def multiply(a, b):\n    return a + b\n",
            test_code=huge_test,
            description="Multiply two numbers.",
            entry_point="multiply",
        )

        assert "Reference test summary:" in prompt
        assert "The full workspace still contains test_buggy.py" in prompt
        assert huge_test not in prompt
        assert len(prompt) < len(huge_test)

    def test_compact_problem_prompt_respects_explicit_max_chars(self) -> None:
        prompt = "A" * 500
        compact = compact_problem_prompt(prompt, max_chars=80)
        assert len(compact) <= 80
        assert compact.endswith("...")


class TestResultExtractor:
    def test_make_diff_produces_valid_diff(self, simple_case: Path) -> None:
        diff = make_diff(simple_case / "buggy.py", "def multiply(a, b):\n    return a * b\n")
        assert diff is not None
        assert "-    return a + b" in diff
        assert "+    return a * b" in diff

    def test_make_diff_identical_returns_none(self, simple_case: Path) -> None:
        same = (simple_case / "buggy.py").read_text(encoding="utf-8")
        assert make_diff(simple_case / "buggy.py", same) is None


class TestTraceCoderAdapter:
    def test_adapter_name(self) -> None:
        assert TraceCoderAdapter().name == "tracecoder"

    def test_setup_raises_on_missing_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            TraceCoderAdapter().setup(BaselineConfig(repo_dir=tmp_path / "missing"))

    def test_setup_requires_llmdebug_patch(self, tmp_path: Path) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_llmdebug_support=False)
        with pytest.raises(RuntimeError, match="llmdebug compatibility support"):
            TraceCoderAdapter().setup(BaselineConfig(repo_dir=repo))

    def test_setup_succeeds_with_valid_repo(self, tmp_path: Path) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))
        assert adapter.repo_commit == ""

    def test_convert_case_delegates(self, simple_case: Path) -> None:
        result = TraceCoderAdapter().convert_case(simple_case)
        assert result.task_id == "my_case"
        assert result.execution_mode == "llmdebug_native"

    def test_run_repair_uses_patched_cli_contract(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=3,
            api_timeout_sec=13,
            timeout_sec=25,
            api_base_url="http://127.0.0.1:9999",
            api_key="sk-test",
            extra={"tracecoder_fake_responses_path": str(tmp_path / "responses.json")},
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)
        observed: dict[str, object] = {}
        observed_row: dict[str, object] = {}

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
            dataset_path = Path(cmd[cmd.index("--dataset-path") + 1])
            output_csv = Path(cmd[cmd.index("-o") + 1])
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            observed_row.update(
                json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
            )
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 3,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": True,
                        "usage_reported_call_count": 3,
                        "usage_missing_call_count": 0,
                        "tokens_prompt": 17,
                        "tokens_completion": 9,
                        "tokens_total": 26,
                        "reported_tokens_prompt": 17,
                        "reported_tokens_completion": 9,
                        "reported_tokens_total": 26,
                        "prompt_chars_total": 121,
                        "completion_chars_total": 58,
                        "chars_total": 179,
                        "reported_chars_prompt": 121,
                        "reported_chars_completion": 58,
                        "reported_chars_total": 179,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "direct_gen_passed": False,
                        "debug_added_value": True,
                        "attempts_completed": 2,
                        "usage_source": "provider_reported",
                    }
                ),
                encoding="utf-8",
            )

            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                        "total_prompt_tokens",
                        "total_completion_tokens",
                        "total_llm_calls",
                        "total_llm_duration_sec",
                        "total_eval_duration_sec",
                        "processing_time_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps(
                            {"passed": False, "passed_count": 0, "total_count": 1}
                        ),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps(
                            {"passed": True, "passed_count": 1, "total_count": 1}
                        ),
                        "final_passed": "True",
                        "debug_session_log": json.dumps([{"attempt": 1}, {"attempt": 2}]),
                        "stopped_reason": "Debug cycle completed",
                        "total_prompt_tokens": "17",
                        "total_completion_tokens": "9",
                        "total_llm_calls": "3",
                        "total_llm_duration_sec": "1.25",
                        "total_eval_duration_sec": "0.75",
                        "processing_time_seconds": "2.50",
                    }
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=12,
                description="Multiply two numbers.",
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "compact_then_patch",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": ["Crash is localized to function multiply"],
                }
            },
            "history_bundle": {
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "keep multiply semantics while changing operator",
                    }
                ]
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["must touch crash file"],
            },
            "solver_hints": {
                "retry_plan": "Adjust only the multiply return operator.",
            },
        }

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.repaired_code == "def multiply(a, b):\n    return a * b\n"
        assert result.iterations_used == 2
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 9,
            "total_tokens": 26,
        }
        assert result.meta["stopped_reason"] == "Debug cycle completed"
        assert result.meta["processing_time_seconds"] == 2.5
        metrics = result.meta["metrics"]
        assert metrics == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": True,
            "usage_reported_call_count": 3,
            "usage_missing_call_count": 0,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 9,
            "reported_tokens_total": 26,
            "prompt_chars_total": 121,
            "completion_chars_total": 58,
            "chars_input": 121,
            "chars_output": 58,
            "chars_total": 179,
            "reported_chars_prompt": 121,
            "reported_chars_completion": 58,
            "reported_chars_total": 179,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": True,
            "usage_source": "provider_reported",
        }

        cmd = observed["cmd"]
        assert isinstance(cmd, list)
        assert cmd[0] == str(config.python_executable)
        assert "-d" in cmd and "llmdebug" in cmd
        assert "--dataset-path" in cmd
        assert "--llmdebug-metrics-path" in cmd
        assert "--max-debug-attempts" in cmd
        assert "--timeout" in cmd
        assert observed["cwd"] == str(repo)
        env = observed["env"]
        assert isinstance(env, dict)
        assert env["LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE"] == "auto"
        assert env["LLMDEBUG_BASELINE_OPENAI_ALLOW_REASONING_FALLBACK"] == "0"
        assert str(Path(__file__).resolve().parents[1]) in env["PYTHONPATH"]
        env = observed["env"]
        assert isinstance(env, dict)
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999"
        assert env["OPENAI_API_KEY"] == "sk-test"
        assert env["TRACECODER_OPENAI_TIMEOUT_SEC"] == "13.0"
        assert env["TRACECODER_FAKE_RESPONSES_PATH"] == str(tmp_path / "responses.json")
        assert env["TRACECODER_ANALYSIS_MAX_TOKENS"] == "4096"
        assert env["TRACECODER_IMPLEMENTATION_MAX_TOKENS"] == "4096"
        assert env["TRACECODER_PROBLEM_SUMMARY_MAX_CHARS"] == "4000"
        assert env["TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS"] == "3000"
        assert env["TRACECODER_FAILURE_FEEDBACK_MAX_CHARS"] == "6000"
        assert env["TRACECODER_FAILURE_FEEDBACK_TAIL_MAX_CHARS"] == "2000"
        assert env["TRACECODER_INSTRUMENTATION_OUTPUT_MAX_LINES"] == "80"
        assert env["TRACECODER_INSTRUMENTATION_OUTPUT_MAX_CHARS"] == "12000"
        assert observed_row["repair_seeded"] is True
        assert observed_row["execution_mode"] == "llmdebug_native"
        assert "LLMDebug controller guidance:" in str(observed_row["complete_prompt"])
        assert "crash_anchor: buggy.py:2 in multiply" in str(observed_row["complete_prompt"])

    def test_run_repair_respects_configured_guidance_and_streak_knobs(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=3,
            extra={
                "baseline_tracecoder_controller_guidance_max_chars": 120,
                "baseline_tracecoder_max_no_improvement_streak": 5,
            },
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)
        observed: dict[str, object] = {}
        observed_row: dict[str, object] = {}

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
            observed["cmd"] = cmd
            dataset_path = Path(cmd[cmd.index("--dataset-path") + 1])
            output_csv = Path(cmd[cmd.index("-o") + 1])
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            observed_row.update(
                json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
            )
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 1,
                        "usage_available": False,
                        "attempts_completed": 1,
                    }
                ),
                encoding="utf-8",
            )
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps({"passed": False}),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps({"passed": True}),
                        "final_passed": "True",
                        "debug_session_log": json.dumps([{"attempt": 1}]),
                        "stopped_reason": "Debug cycle completed",
                    }
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=12,
                description="Multiply two numbers.",
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "compact_then_patch",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": ["A" * 300, "B" * 300],
                },
            },
            "history_bundle": {
                "schema_version": "v1",
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "keep multiply semantics while changing operator",
                    }
                ],
            },
            "verifier_report": {"schema_version": "v1"},
            "solver_hints": {
                "retry_plan": "Adjust only the multiply return operator.",
            },
        }

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        cmd = observed["cmd"]
        assert isinstance(cmd, list)
        assert cmd[cmd.index("--max-no-improvement-streak") + 1] == "5"
        complete_prompt = str(observed_row["complete_prompt"])
        assert "LLMDebug controller guidance:" in complete_prompt
        guidance_suffix = complete_prompt.split("LLMDebug controller guidance:\n", 1)[1]
        assert len(guidance_suffix) <= 120

    def test_run_repair_uses_fair_v1_hybrid_guidance_block(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            extra={
                "baseline_hybrid_guidance_mode": "fair_v1",
                "baseline_tracecoder_controller_guidance_max_chars": 600,
            },
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)
        observed_row: dict[str, object] = {}

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
            dataset_path = Path(cmd[cmd.index("--dataset-path") + 1])
            output_csv = Path(cmd[cmd.index("-o") + 1])
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            observed_row.update(
                json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
            )
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 1,
                        "usage_available": False,
                        "attempts_completed": 1,
                    }
                ),
                encoding="utf-8",
            )
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps({"passed": False}),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps({"passed": True}),
                        "final_passed": "True",
                        "debug_session_log": json.dumps([{"attempt": 1}]),
                        "stopped_reason": "Debug cycle completed",
                    }
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=12,
                description="Multiply two numbers.",
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "compact_then_patch",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": ["Focus on the operator", "Keep multiply semantics stable"],
                },
            },
            "history_bundle": {
                "schema_version": "v1",
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "keep multiply semantics while changing operator",
                    }
                ],
            },
            "verifier_report": {
                "schema_version": "v1",
                "status": "previous_verifier_rejected",
                "reasons": ["touch buggy.py", "preserve signature"],
            },
            "solver_hints": {
                "retry_plan": "Adjust only the multiply return operator.",
                "strategy_summary": "should not appear",
                "decomposition": [{"title": "localize", "question": "should not appear"}],
            },
        }

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        complete_prompt = str(observed_row["complete_prompt"])
        assert "LLMDebug controller guidance:" in complete_prompt
        assert "verifier: status=previous_verifier_rejected" in complete_prompt
        assert "retry_plan: Adjust only the multiply return operator." in complete_prompt
        assert "strategy_summary:" not in complete_prompt
        assert "decomposition:" not in complete_prompt

    def test_run_repair_emits_sidecar_progress_updates(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)
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
            output_csv = Path(cmd[cmd.index("-o") + 1])
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "status": "running",
                        "llm_call_count": 2,
                        "eval_duration_sec": 0.5,
                        "attempts_completed": 1,
                        "stopped_reason": "Debug cycle completed",
                        "direct_gen_passed": False,
                    }
                ),
                encoding="utf-8",
            )
            payload = state_supplier()
            assert payload is not None
            progress_callback(progress_event, payload)
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps({"passed": False}),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps({"passed": True}),
                        "final_passed": "True",
                        "debug_session_log": json.dumps([{"attempt": 1}]),
                        "stopped_reason": "Debug cycle completed",
                    }
                )
            return MonitoredSubprocessResult(
                returncode=0,
                duration_sec=0.05,
                stdout="",
                stderr="",
            )

        monkeypatch.setattr(
            "baselines.tracecoder.adapter.run_monitored_subprocess",
            _fake_monitored_run,
        )

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is True
        progress_payloads = [payload for event, payload in events if event == "baseline_progress"]
        assert progress_payloads
        assert progress_payloads[-1] == {
            "owner": "baseline",
            "source": "sidecar",
            "phase_label": "baseline: tracecoder subprocess",
            "waiting_for": "tracecoder_subprocess",
            "external_operation": "baseline_repair",
            "external_provider": "tracecoder",
            "attempts_completed": 1,
            "llm_calls_observed": 2,
            "eval_duration_sec_observed": 0.5,
            "note": "status=running stopped=Debug cycle completed direct_gen=false",
        }

    def test_run_repair_progress_falls_back_to_stdout_excerpt(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)
        events: list[tuple[str, dict[str, object] | None]] = []
        adapter.set_progress_callback(lambda event, payload: events.append((event, payload)))

        def _fake_monitored_run(
            *,
            cmd: list[str],
            progress_callback,
            progress_event: str,
            progress_base_payload,
            state_supplier,
            **_kwargs,
        ) -> MonitoredSubprocessResult:
            output_csv = Path(cmd[cmd.index("-o") + 1])
            assert state_supplier() is None
            progress_callback(
                progress_event,
                {
                    **dict(progress_base_payload),
                    "source": "stdout",
                    "activity_excerpt": "tracecoder debug cycle checkpoint",
                },
            )
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps({"passed": False}),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps({"passed": True}),
                        "final_passed": "True",
                        "debug_session_log": json.dumps([{"attempt": 1}]),
                        "stopped_reason": "Debug cycle completed",
                    }
                )
            return MonitoredSubprocessResult(
                returncode=0,
                duration_sec=0.05,
                stdout="tracecoder debug cycle checkpoint\n",
                stderr="",
            )

        monkeypatch.setattr(
            "baselines.tracecoder.adapter.run_monitored_subprocess",
            _fake_monitored_run,
        )

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is True
        progress_payloads = [payload for event, payload in events if event == "baseline_progress"]
        assert progress_payloads
        assert progress_payloads[-1] == {
            "owner": "baseline",
            "phase_label": "baseline: tracecoder subprocess",
            "waiting_for": "tracecoder_subprocess",
            "external_operation": "baseline_repair",
            "external_provider": "tracecoder",
            "source": "stdout",
            "activity_excerpt": "tracecoder debug cycle checkpoint",
        }

    def test_run_repair_drops_csv_usage_when_sidecar_marks_usage_partial(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=3,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

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
            output_csv = Path(cmd[cmd.index("-o") + 1])
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 3,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": False,
                        "usage_reported_call_count": 2,
                        "usage_missing_call_count": 1,
                        "reported_tokens_prompt": 17,
                        "reported_tokens_completion": 9,
                        "reported_tokens_total": 26,
                        "prompt_chars_total": 121,
                        "completion_chars_total": 58,
                        "chars_total": 179,
                        "reported_chars_prompt": 121,
                        "reported_chars_completion": 58,
                        "reported_chars_total": 179,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "direct_gen_passed": False,
                        "debug_added_value": True,
                        "attempts_completed": 2,
                        "usage_source": "partial_provider_reported",
                    }
                ),
                encoding="utf-8",
            )
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                        "total_prompt_tokens",
                        "total_completion_tokens",
                        "total_llm_calls",
                        "total_llm_duration_sec",
                        "total_eval_duration_sec",
                        "processing_time_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps(
                            {"passed": False, "passed_count": 0, "total_count": 1}
                        ),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps(
                            {"passed": True, "passed_count": 1, "total_count": 1}
                        ),
                        "final_passed": "True",
                        "debug_session_log": json.dumps([{"attempt": 1}, {"attempt": 2}]),
                        "stopped_reason": "Debug cycle completed",
                        "total_prompt_tokens": "17",
                        "total_completion_tokens": "9",
                        "total_llm_calls": "3",
                        "total_llm_duration_sec": "1.25",
                        "total_eval_duration_sec": "0.75",
                        "processing_time_seconds": "2.50",
                    }
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is True
        assert result.iterations_used == 2
        assert result.token_usage == {}
        assert result.meta["metrics"] == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 9,
            "reported_tokens_total": 26,
            "prompt_chars_total": 121,
            "completion_chars_total": 58,
            "chars_input": 121,
            "chars_output": 58,
            "chars_total": 179,
            "reported_chars_prompt": 121,
            "reported_chars_completion": 58,
            "reported_chars_total": 179,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": True,
            "usage_source": "partial_provider_reported",
        }

    def test_parse_output_missing_csv(self, tmp_path: Path) -> None:
        result = TraceCoderAdapter()._parse_output(
            output_csv=tmp_path / "missing.csv",
            stdout="",
            stderr="",
            returncode=1,
        )
        assert result.success is False
        assert result.meta.get("error") == "missing_output_csv"

    def test_parse_output_rejects_missing_required_diagnostics(self, tmp_path: Path) -> None:
        output_csv = tmp_path / "output.csv"
        with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "final_code",
                    "final_passed",
                    "processing_time_seconds",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "final_code": "def multiply(a, b):\n    return a * b\n",
                    "final_passed": "True",
                    "processing_time_seconds": "1.0",
                }
            )

        result = TraceCoderAdapter()._parse_output(
            output_csv=output_csv,
            stdout="",
            stderr="",
            returncode=0,
        )

        assert result.success is False
        assert result.repaired_code is None
        assert result.meta["error"] == "baseline_incompatible_output_schema"
        assert result.meta["schema_missing_fields"] == [
            "method_instrumentation",
            "method_two_step_repair",
            "direct_gen_passed",
            "stopped_reason",
        ]

    def test_parse_output_surfaces_openai_compat_reasoning_only_error(self, tmp_path: Path) -> None:
        output_csv = tmp_path / "output.csv"
        metrics_sidecar_path = tmp_path / "output.csv.llmdebug_metrics.json"
        with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task_id",
                    "model",
                    "dataset",
                    "method_instrumentation",
                    "method_two_step_repair",
                    "direct_gen_eval",
                    "direct_gen_passed",
                    "final_code",
                    "final_eval",
                    "final_passed",
                    "debug_session_log",
                    "stopped_reason",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "task_id": "my_case",
                    "model": "test-model",
                    "dataset": "llmdebug",
                    "method_instrumentation": "True",
                    "method_two_step_repair": "True",
                    "direct_gen_eval": json.dumps({"passed": False}),
                    "direct_gen_passed": "False",
                    "final_code": "",
                    "final_eval": json.dumps({"passed": False}),
                    "final_passed": "False",
                    "debug_session_log": json.dumps([]),
                    "stopped_reason": "OpenAI-compatible server returned reasoning without final content during direct generation",
                }
            )
        metrics_sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "baseline": "tracecoder",
                    "last_llm_failure_kind": "openai_compat_reasoning_only",
                    "last_llm_failure_stage": "direct_generation",
                    "compat_message": "Disable thinking in the local model template.",
                    "compat_profile_applied": "lmstudio_model_yaml",
                    "empty_content_with_reasoning": True,
                    "reasoning_present": True,
                }
            ),
            encoding="utf-8",
        )

        result = TraceCoderAdapter()._parse_output(
            output_csv=output_csv,
            metrics_sidecar_path=metrics_sidecar_path,
            stdout="",
            stderr="",
            returncode=0,
        )

        assert result.success is False
        assert result.meta["error"] == "openai_compat_reasoning_only"
        assert result.meta["compat_message"] == "Disable thinking in the local model template."
        assert result.meta["compat_meta"]["compat_profile_applied"] == "lmstudio_model_yaml"

    def test_normalize_failed_unchanged_candidate_clears_repaired_code(self) -> None:
        adapter = TraceCoderAdapter()

        result = adapter._normalize_failed_unchanged_candidate(
            BaselineResult(
                repaired_code="def multiply(a, b):\n    return a + b\n",
                success=False,
                meta={"stopped_reason": "No repair plan produced"},
            ),
            original_buggy_code="def multiply(a, b):\n    return a + b\n",
        )

        assert result.success is False
        assert result.repaired_code is None
        assert result.meta["candidate_status"] == "unchanged_original_code"
        assert result.meta["artifact_error"] == "unchanged_original_code"
        assert result.meta["error"] == "unchanged_original_code"

    def test_run_repair_timeout_marks_usage_unavailable_without_csv(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(
                cmd=cmd, timeout=timeout, output="partial out", stderr="partial err"
            )

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.token_usage == {}
        assert result.meta["error"] == "timeout"
        assert result.meta["timeout_sec"] == 25
        assert result.meta["stdout_tail"] == "partial out"
        assert result.meta["stderr_tail"] == "partial err"
        assert result.meta["metrics"] == {
            "llm_call_count": 0,
            "usage_available": False,
            "usage_source": "not_reported",
        }

    def test_run_repair_timeout_recovers_exact_usage_from_sidecar_without_csv(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 3,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": True,
                        "usage_reported_call_count": 3,
                        "usage_missing_call_count": 0,
                        "tokens_prompt": 17,
                        "tokens_completion": 9,
                        "tokens_total": 26,
                        "reported_tokens_prompt": 17,
                        "reported_tokens_completion": 9,
                        "reported_tokens_total": 26,
                        "prompt_chars_total": 121,
                        "completion_chars_total": 58,
                        "chars_total": 179,
                        "reported_chars_prompt": 121,
                        "reported_chars_completion": 58,
                        "reported_chars_total": 179,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "direct_gen_passed": False,
                        "debug_added_value": False,
                        "attempts_completed": 1,
                        "usage_source": "provider_reported",
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(
                cmd=cmd, timeout=timeout, output="partial out", stderr="partial err"
            )

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.iterations_used == 1
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 9,
            "total_tokens": 26,
        }
        assert result.meta["error"] == "timeout"
        assert result.meta["metrics"] == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": True,
            "usage_reported_call_count": 3,
            "usage_missing_call_count": 0,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 9,
            "reported_tokens_total": 26,
            "prompt_chars_total": 121,
            "completion_chars_total": 58,
            "chars_input": 121,
            "chars_output": 58,
            "chars_total": 179,
            "reported_chars_prompt": 121,
            "reported_chars_completion": 58,
            "reported_chars_total": 179,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": False,
            "usage_source": "provider_reported",
        }

    def test_run_repair_timeout_recovers_partial_usage_from_sidecar_without_csv(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 3,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": False,
                        "usage_reported_call_count": 2,
                        "usage_missing_call_count": 1,
                        "reported_tokens_prompt": 17,
                        "reported_tokens_completion": 9,
                        "reported_tokens_total": 26,
                        "prompt_chars_total": 121,
                        "completion_chars_total": 58,
                        "chars_total": 179,
                        "reported_chars_prompt": 121,
                        "reported_chars_completion": 58,
                        "reported_chars_total": 179,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "direct_gen_passed": False,
                        "debug_added_value": False,
                        "attempts_completed": 1,
                        "usage_source": "partial_provider_reported",
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.iterations_used == 1
        assert result.token_usage == {}
        assert result.meta["error"] == "timeout"
        assert result.meta["metrics"] == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 9,
            "reported_tokens_total": 26,
            "prompt_chars_total": 121,
            "completion_chars_total": 58,
            "chars_input": 121,
            "chars_output": 58,
            "chars_total": 179,
            "reported_chars_prompt": 121,
            "reported_chars_completion": 58,
            "reported_chars_total": 179,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": False,
            "usage_source": "partial_provider_reported",
        }

    def test_run_repair_timeout_preserves_timeout_failure_metadata_from_sidecar(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            api_timeout_sec=13,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 1,
                        "llm_duration_sec": 65.0,
                        "eval_duration_sec": 0.0,
                        "internal_total_duration_sec": 65.0,
                        "orchestration_duration_sec": 0.0,
                        "usage_available": False,
                        "usage_reported_call_count": 0,
                        "usage_missing_call_count": 1,
                        "attempts_completed": 0,
                        "usage_source": "not_reported",
                        "llm_timeout_count": 1,
                        "last_llm_failure_kind": "api_timeout",
                        "last_llm_failure_stage": "analysis_planning",
                        "stopped_reason": "LLM request timed out during analysis planning",
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.iterations_used == 0
        assert result.meta["error"] == "timeout"
        assert result.meta["metrics"] == {
            "llm_call_count": 1,
            "llm_duration_sec": 65.0,
            "eval_duration_sec": 0.0,
            "internal_total_duration_sec": 65.0,
            "orchestration_duration_sec": 0.0,
            "usage_available": False,
            "usage_reported_call_count": 0,
            "usage_missing_call_count": 1,
            "usage_source": "not_reported",
            "llm_timeout_count": 1,
            "last_llm_failure_kind": "api_timeout",
            "last_llm_failure_stage": "analysis_planning",
            "stopped_reason": "LLM request timed out during analysis planning",
        }

    def test_run_repair_timeout_preserves_not_reported_usage_from_sidecar(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 3,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": False,
                        "usage_reported_call_count": 0,
                        "usage_missing_call_count": 3,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "direct_gen_passed": False,
                        "debug_added_value": False,
                        "attempts_completed": 1,
                        "usage_source": "not_reported",
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.iterations_used == 1
        assert result.token_usage == {}
        assert result.meta["error"] == "timeout"
        assert result.meta["metrics"] == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": False,
            "usage_reported_call_count": 0,
            "usage_missing_call_count": 3,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": False,
            "usage_source": "not_reported",
        }

    def test_run_repair_timeout_preserves_timeout_on_partial_schema_csv(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            output_csv = Path(cmd[cmd.index("-o") + 1])
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "final_code",
                        "final_passed",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_passed": "False",
                    }
                )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.meta["error"] == "timeout"
        assert result.meta["timeout_sec"] == 25
        assert result.meta["timeout_parse_error"] == "baseline_incompatible_output_schema"
        assert result.meta["metrics"] == {
            "llm_call_count": 0,
            "usage_available": False,
        }

    def test_run_repair_timeout_preserves_usage_from_partial_csv(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            output_csv = Path(cmd[cmd.index("-o") + 1])
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                        "total_prompt_tokens",
                        "total_completion_tokens",
                        "total_llm_calls",
                        "total_llm_duration_sec",
                        "total_eval_duration_sec",
                        "processing_time_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps(
                            {"passed": False, "passed_count": 0, "total_count": 1}
                        ),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps(
                            {"passed": False, "passed_count": 0, "total_count": 1}
                        ),
                        "final_passed": "False",
                        "debug_session_log": json.dumps([{"attempt": 1}]),
                        "stopped_reason": "Debug cycle completed",
                        "total_prompt_tokens": "17",
                        "total_completion_tokens": "9",
                        "total_llm_calls": "3",
                        "total_llm_duration_sec": "1.25",
                        "total_eval_duration_sec": "0.75",
                        "processing_time_seconds": "2.50",
                    }
                )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.iterations_used == 1
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 9,
            "total_tokens": 26,
        }
        assert result.meta["error"] == "timeout"
        assert result.meta["timeout_sec"] == 25
        assert result.meta["metrics"] == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": True,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": False,
            "usage_source": "provider_reported",
        }

    def test_run_repair_timeout_drops_csv_usage_when_sidecar_marks_usage_partial(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_tracecoder_repo(tmp_path, with_metrics_sidecar_support=True)
        adapter = TraceCoderAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=tmp_path / "fake-python",
            model="test-model",
            max_attempts=2,
            timeout_sec=25,
        )
        config.python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        config.python_executable.chmod(0o755)
        adapter.setup(config)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: float,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, cwd, env
            output_csv = Path(cmd[cmd.index("-o") + 1])
            metrics_sidecar_path = Path(cmd[cmd.index("--llmdebug-metrics-path") + 1])
            metrics_sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "baseline": "tracecoder",
                        "llm_call_count": 3,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": False,
                        "usage_reported_call_count": 2,
                        "usage_missing_call_count": 1,
                        "reported_tokens_prompt": 17,
                        "reported_tokens_completion": 9,
                        "reported_tokens_total": 26,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "direct_gen_passed": False,
                        "debug_added_value": False,
                        "attempts_completed": 1,
                        "usage_source": "partial_provider_reported",
                    }
                ),
                encoding="utf-8",
            )
            with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "dataset",
                        "method_instrumentation",
                        "method_two_step_repair",
                        "direct_gen_eval",
                        "direct_gen_passed",
                        "final_code",
                        "final_eval",
                        "final_passed",
                        "debug_session_log",
                        "stopped_reason",
                        "total_prompt_tokens",
                        "total_completion_tokens",
                        "total_llm_calls",
                        "total_llm_duration_sec",
                        "total_eval_duration_sec",
                        "processing_time_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "my_case",
                        "model": "test-model",
                        "dataset": "llmdebug",
                        "method_instrumentation": "True",
                        "method_two_step_repair": "True",
                        "direct_gen_eval": json.dumps(
                            {"passed": False, "passed_count": 0, "total_count": 1}
                        ),
                        "direct_gen_passed": "False",
                        "final_code": "def multiply(a, b):\n    return a * b\n",
                        "final_eval": json.dumps(
                            {"passed": False, "passed_count": 0, "total_count": 1}
                        ),
                        "final_passed": "False",
                        "debug_session_log": json.dumps([{"attempt": 1}]),
                        "stopped_reason": "Debug cycle completed",
                        "total_prompt_tokens": "17",
                        "total_completion_tokens": "9",
                        "total_llm_calls": "3",
                        "total_llm_duration_sec": "1.25",
                        "total_eval_duration_sec": "0.75",
                        "processing_time_seconds": "2.50",
                    }
                )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="")

        _patch_monitored_run(monkeypatch, _fake_run)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.iterations_used == 1
        assert result.token_usage == {}
        assert result.meta["error"] == "timeout"
        assert result.meta["metrics"] == {
            "llm_call_count": 3,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 9,
            "reported_tokens_total": 26,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": True,
            "direct_gen_passed": False,
            "debug_added_value": False,
            "usage_source": "partial_provider_reported",
        }

    def test_extract_diff_with_repair(self, simple_case: Path) -> None:
        result = BaselineResult(
            repaired_code="def multiply(a, b):\n    return a * b\n",
            success=True,
            iterations_used=1,
        )
        diff = TraceCoderAdapter().extract_diff(simple_case, result)
        assert diff is not None
        assert "+    return a * b" in diff

    def test_metrics_patch_uses_final_pass_for_debug_added_value(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert (
            '"debug_added_value": result["final_passed"] and not result["direct_gen_passed"]'
            in patch_text
        )
        assert 'result["final_eval"].get("passed_count", 0)' not in patch_text

    def test_metrics_patch_tracks_chars_and_usage_coverage(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert '"prompt_chars_total": prompt_chars_total' in patch_text
        assert '"completion_chars_total": completion_chars_total' in patch_text
        assert '"usage_missing_call_count": usage_missing_call_count' in patch_text
        assert "partial_provider_reported" in patch_text

    def test_generation_patch_accepts_input_output_usage_fields(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert 'usage.get("input_tokens")' in patch_text
        assert 'usage.get("output_tokens")' in patch_text

    def test_generation_patch_wires_explicit_api_timeout(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert 'os.getenv("TRACECODER_OPENAI_TIMEOUT_SEC", "").strip()' in patch_text
        assert 'client_kwargs["timeout"] = timeout_sec' in patch_text
        assert "Error: API timeout after" in patch_text

    def test_generation_patch_wires_stage_specific_token_caps(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert "TRACECODER_ANALYSIS_MAX_TOKENS" in patch_text
        assert "TRACECODER_IMPLEMENTATION_MAX_TOKENS" in patch_text

    def test_metrics_patch_adds_llmdebug_dataset_evaluator_and_progress_module(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert "+++ b/datasets/llmdebug/evaluation.py" in patch_text
        assert "+++ b/src/progress.py" in patch_text
        assert "run_llmdebug_pytest_case" in patch_text

    def test_metrics_patch_adds_exact_test_context_and_failure_feedback_support(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert "+++ b/src/exact_test_context.py" in patch_text
        assert "build_exact_test_context" in patch_text
        assert "exact_test_context_meta" in patch_text
        assert "build_llmdebug_failure_feedback" in patch_text
        assert "TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS" in patch_text
        assert "TRACECODER_FAILURE_FEEDBACK_MAX_CHARS" in patch_text
        assert "TRACECODER_INSTRUMENTATION_OUTPUT_MAX_CHARS" in patch_text

    def test_problem_processor_patch_tracks_timeout_failure_metadata(self) -> None:
        patch_text = _TRACECODER_METRICS_PATCH.read_text(encoding="utf-8")

        assert '"llm_timeout_count": total_llm_timeout_count' in patch_text
        assert '"last_llm_failure_kind": last_llm_failure_kind' in patch_text
        assert '"last_llm_failure_stage": last_llm_failure_stage' in patch_text
        assert "def _llm_failure_stop_reason" in patch_text
        assert '"analysis_planning": "analysis planning"' in patch_text


# ===========================================================================
# Regression tests for TraceCoder code extraction improvements
# ===========================================================================


class TestParseImplementationOutput:
    """Tests for the improved _parse_implementation_output in problem_processor.py.

    The function lives in the vendored TraceCoder repo. Since importing the
    full module pulls heavy dependencies (pandas), we extract and test the
    function logic inline, then verify the vendored copy is in sync.
    """

    @staticmethod
    def _parse(response_text: str) -> str:
        """Reimplementation matching vendored _parse_implementation_output."""
        import re

        # Strategy 1: CODE: block with ```python
        m = re.search(r"CODE:\s*```python\n(.*?)\n```", response_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Strategy 2: CODE: with ``` without language tag
        m = re.search(r"CODE:\s*```\n(.*?)\n```", response_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Strategy 3: CODE: flexible whitespace
        m = re.search(r"CODE:\s*\n*\s*```(?:python)?\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Strategy 4: Everything after CODE: marker
        m = re.search(r"CODE:(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if m:
            potential = m.group(1).strip()
            if potential.startswith("```python"):
                potential = potential[len("```python") :].strip()
            if potential.startswith("```"):
                potential = potential[len("```") :].strip()
            if potential.endswith("```"):
                potential = potential[: -len("```")].strip()
            end_pos = potential.find("END_CODE")
            if end_pos >= 0:
                potential = potential[:end_pos].strip()
            if potential:
                return potential
        # Strategy 5: Last ```python block
        blocks = list(re.finditer(r"```python\s*\n(.*?)\n\s*```", response_text, re.DOTALL))
        if blocks:
            candidate = blocks[-1].group(1).strip()
            if candidate:
                return candidate
        # Strategy 6: Last ``` block with Python structure
        blocks = list(re.finditer(r"```\s*\n(.*?)\n\s*```", response_text, re.DOTALL))
        if blocks:
            candidate = blocks[-1].group(1).strip()
            if candidate and re.search(
                r"^\s*(def |class |import |from |[a-zA-Z_]\w*\s*=)", candidate, re.MULTILINE
            ):
                return candidate
        return ""

    def test_vendored_function_in_sync(self) -> None:
        """Verify the vendored problem_processor.py contains our strategies."""
        pp = (
            Path(__file__).resolve().parents[1]
            / "baselines"
            / "repos"
            / "TraceCoder"
            / "problem_processor.py"
        )
        if not pp.is_file():
            pytest.skip("real TraceCoder repo not provisioned")
        text = pp.read_text(encoding="utf-8")
        assert "def _parse_implementation_output" in text
        # Check that our new fallback strategies are present
        assert "Strategy 5" in text or "Last ```python block" in text or "fenced_blocks" in text
        assert "Strategy 6" in text or "without language tag" in text or "Python structure" in text

    def test_strict_code_format(self) -> None:
        response = "CODE:\n```python\ndef multiply(a, b):\n    return a * b\n```"
        result = self._parse(response)
        assert "return a * b" in result

    def test_code_with_extra_whitespace(self) -> None:
        response = "CODE: \n\n ```python\ndef f():\n    return 42\n ```"
        result = self._parse(response)
        assert "return 42" in result

    def test_code_marker_without_fence(self) -> None:
        response = "CODE:\ndef multiply(a, b):\n    return a * b\n"
        result = self._parse(response)
        assert "return a * b" in result

    def test_no_code_marker_but_fenced_python_block(self) -> None:
        """Models sometimes skip the CODE: marker but emit fenced Python."""
        response = (
            "Here is the corrected code:\n\n```python\ndef multiply(a, b):\n    return a * b\n```\n"
        )
        result = self._parse(response)
        assert "return a * b" in result

    def test_multiple_fenced_blocks_picks_last(self) -> None:
        """When multiple ```python blocks exist, prefer the last (complete code)."""
        response = (
            "Repaired CODE Part:\n"
            "```python\n"
            "    return a * b\n"
            "```\n\n"
            "```python\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "```\n"
        )
        result = self._parse(response)
        assert "def multiply" in result
        assert "return a * b" in result

    def test_end_code_marker_stripped(self) -> None:
        response = "CODE:\n```python\ndef f():\n    return 1\n```\nEND_CODE\n"
        result = self._parse(response)
        assert "return 1" in result
        assert "END_CODE" not in result

    def test_empty_response_returns_empty(self) -> None:
        assert self._parse("") == ""

    def test_prose_only_returns_empty(self) -> None:
        response = "The bug is in the multiply function. You need to change + to *."
        result = self._parse(response)
        assert result == ""

    def test_unfenced_code_block_with_python_structure(self) -> None:
        """Bare ``` blocks with Python-like content should be accepted."""
        response = "Here is the fix:\n\n```\ndef multiply(a, b):\n    return a * b\n```\n"
        result = self._parse(response)
        assert "return a * b" in result
