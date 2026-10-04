"""Tests for the standalone upstream RLM baseline adapter."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from baselines.adapter_base import BaselineCaseContext, BaselineConfig
from baselines.rlm import env_client
from baselines.rlm.adapter import RLMAdapter, _CommandOutcome
from baselines.rlm.case_converter import convert_case
from llmdebug import _json as json


@pytest.fixture()
def rlm_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "rlm_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text(
        "def offset(value):\n    return value * 2\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import compute\n\n\ndef test_compute():\n    assert compute(3) == 6\n",
        encoding="utf-8",
    )
    return case_dir


def _make_rlm_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rlm"
    (repo / "rlm" / "core").mkdir(parents=True)
    (repo / "rlm" / "clients").mkdir(parents=True)
    (repo / "rlm" / "environments").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='rlms'\n", encoding="utf-8")
    (repo / "rlm" / "__init__.py").write_text("from rlm.core.rlm import RLM\n", encoding="utf-8")
    (repo / "rlm" / "core" / "rlm.py").write_text(
        "class RLM:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n"
        "    def completion(self, prompt, root_prompt=None):\n"
        "        raise RuntimeError('stub')\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "__init__.py").write_text(
        "def get_client(backend, backend_kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "openai.py").write_text(
        "class OpenAIClient:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.model_name = kwargs.get('model_name', '')\n"
        "    def completion(self, prompt, model=None):\n"
        "        return ''\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "environments" / "local_repl.py").write_text(
        "class LocalREPL:\n    pass\n",
        encoding="utf-8",
    )
    return repo


def _make_env_client_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rlm"
    (repo / "rlm" / "utils").mkdir(parents=True)
    (repo / "rlm" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "rlm" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "rlm" / "utils" / "exceptions.py").write_text(
        "class BudgetExceededError(Exception):\n    pass\n"
        "class CancellationError(Exception):\n    partial_answer = ''\n"
        "class ErrorThresholdExceededError(Exception):\n    partial_answer = ''\n"
        "class TimeoutExceededError(Exception):\n    partial_answer = ''\n"
        "class TokenLimitExceededError(Exception):\n    partial_answer = ''\n",
        encoding="utf-8",
    )
    return repo


def _make_env_client_tracking_repo(tmp_path: Path) -> Path:
    repo = _make_env_client_repo(tmp_path)
    (repo / "rlm" / "clients").mkdir(parents=True)
    (repo / "rlm" / "core").mkdir(parents=True)
    (repo / "rlm" / "environments").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='rlms'\n", encoding="utf-8")
    (repo / "rlm" / "__init__.py").write_text("from rlm.core.rlm import RLM\n", encoding="utf-8")
    (repo / "rlm" / "clients" / "__init__.py").write_text(
        "from rlm.clients.openai import OpenAIClient\n\n"
        "def get_client(backend, backend_kwargs):\n"
        "    if backend != 'openai':\n"
        "        raise ValueError(f'unsupported backend: {backend}')\n"
        "    return OpenAIClient(**backend_kwargs)\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "openai.py").write_text(
        "import openai\n\n"
        "DEFAULT_PRIME_INTELLECT_BASE_URL = ''\n\n"
        "class OpenAIClient:\n"
        "    def __init__(self, model_name='', api_key='', base_url=None, timeout=None):\n"
        "        self.model_name = model_name\n"
        "        self.base_url = base_url\n"
        "        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)\n"
        "        self.async_client = openai.AsyncOpenAI(\n"
        "            api_key=api_key,\n"
        "            base_url=base_url,\n"
        "            timeout=timeout,\n"
        "        )\n"
        "    def _track_cost(self, response, model):\n"
        "        del response, model\n"
        "        return None\n"
        "    def get_usage_summary(self):\n"
        "        return {'model_usage_summaries': {}}\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "core" / "rlm.py").write_text(
        "from rlm.clients import get_client\n\n"
        "class _Completion:\n"
        "    def __init__(self, response, root_model):\n"
        "        self.response = response\n"
        "        self.root_model = root_model\n"
        "        self.metadata = {'iterations': []}\n\n"
        "class RLM:\n"
        "    def __init__(self, *, backend, backend_kwargs, logger=None, **kwargs):\n"
        "        del kwargs\n"
        "        self.backend = backend\n"
        "        self.backend_kwargs = backend_kwargs\n"
        "        self.logger = logger\n"
        "    def completion(self, prompt, root_prompt=None):\n"
        "        del root_prompt\n"
        "        client = get_client(self.backend, self.backend_kwargs)\n"
        "        response = client.completion([\n"
        "            {'role': 'user', 'content': prompt},\n"
        "            {'role': 'assistant', 'content': 'prefill'},\n"
        "        ])\n"
        "        return _Completion(response, self.backend_kwargs.get('model_name', ''))\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "logger.py").write_text(
        "class RLMLogger:\n    def get_trajectory(self):\n        return {'iterations': []}\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "environments" / "local_repl.py").write_text(
        "class LocalREPL:\n    pass\n",
        encoding="utf-8",
    )
    return repo


def _make_env_client_default_answer_repo(tmp_path: Path) -> Path:
    repo = _make_env_client_tracking_repo(tmp_path)
    (repo / "rlm" / "core" / "rlm.py").write_text(
        "from rlm.clients import get_client\n\n"
        "class _Completion:\n"
        "    def __init__(self, response, root_model):\n"
        "        self.response = response\n"
        "        self.root_model = root_model\n"
        "        self.metadata = {'iterations': []}\n\n"
        "class RLM:\n"
        "    def __init__(self, *, backend, backend_kwargs, logger=None, **kwargs):\n"
        "        del kwargs\n"
        "        self.backend = backend\n"
        "        self.backend_kwargs = backend_kwargs\n"
        "        self.logger = logger\n"
        "    def _default_answer(self, message_history, lm_handler):\n"
        "        del message_history, lm_handler\n"
        "        return 'GENERIC DEFAULT'\n"
        "    def completion(self, prompt, root_prompt=None):\n"
        "        del prompt, root_prompt\n"
        "        client = get_client(self.backend, self.backend_kwargs)\n"
        "        response = self._default_answer([\n"
        "            {'role': 'user', 'content': 'repair prompt'},\n"
        "        ], client)\n"
        "        return _Completion(response, self.backend_kwargs.get('model_name', ''))\n",
        encoding="utf-8",
    )
    return repo


def _make_env_client_child_default_answer_repo(tmp_path: Path) -> Path:
    repo = _make_env_client_tracking_repo(tmp_path)
    (repo / "rlm" / "core" / "rlm.py").write_text(
        "from rlm.clients import get_client\n\n"
        "class _Completion:\n"
        "    def __init__(self, response, root_model):\n"
        "        self.response = response\n"
        "        self.root_model = root_model\n"
        "        self.metadata = {'iterations': []}\n\n"
        "class RLM:\n"
        "    def __init__(self, *, backend, backend_kwargs, logger=None, **kwargs):\n"
        "        del kwargs\n"
        "        self.backend = backend\n"
        "        self.backend_kwargs = backend_kwargs\n"
        "        self.logger = logger\n"
        "    def _default_answer(self, message_history, lm_handler):\n"
        "        del message_history, lm_handler\n"
        "        return 'GENERIC DEFAULT'\n"
        "    def completion(self, prompt, root_prompt=None):\n"
        "        del root_prompt\n"
        "        client = get_client(self.backend, self.backend_kwargs)\n"
        "        if prompt == 'ctx':\n"
        "            child = type(self)(\n"
        "                backend=self.backend,\n"
        "                backend_kwargs=self.backend_kwargs,\n"
        "                logger=self.logger,\n"
        "            )\n"
        "            response = child.completion('child prompt').response\n"
        "        else:\n"
        "            response = self._default_answer([\n"
        "                {'role': 'user', 'content': prompt},\n"
        "            ], client)\n"
        "        return _Completion(response, self.backend_kwargs.get('model_name', ''))\n",
        encoding="utf-8",
    )
    return repo


class TestRLMCaseConverter:
    def test_convert_case_packages_prompt_context_and_files(self, rlm_case: Path) -> None:
        context = BaselineCaseContext(
            case_id="rlm_case",
            python_args=("-m", "pytest", "-q"),
            timeout_sec=12,
        )

        result = convert_case(rlm_case, case_context=context)

        assert result.task_id == "rlm_case"
        assert result.entry_point == "compute"
        assert result.python_args == ["-m", "pytest", "-q"]
        assert result.timeout_sec == 12
        assert "Primary symbol: compute" in str(result.prompt_context)
        paths = {row["path"] for row in result.files}
        assert paths == {"buggy.py", "helper.py", "test_buggy.py"}

    def test_convert_case_prefers_case_context_case_id(self, rlm_case: Path) -> None:
        result = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="canonical_case_id",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=12,
            ),
        )

        assert result.task_id == "canonical_case_id"


class TestRLMAdapter:
    def test_adapter_name(self) -> None:
        assert RLMAdapter().name == "rlm"

    def test_setup_raises_on_missing_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            RLMAdapter().setup(BaselineConfig(repo_dir=tmp_path / "missing"))

    def test_setup_succeeds_with_minimal_repo(self, tmp_path: Path) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))
        assert adapter.repo_commit == ""

    def test_setup_rejects_stale_env_client_compatibility_declaration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        monkeypatch.setattr("baselines.rlm.adapter.rlm_env_client._COMPATIBLE_WITH", "rlm@deadbeef")

        with pytest.raises(RuntimeError, match="compatibility declaration is stale"):
            adapter.setup(BaselineConfig(repo_dir=repo))

    def test_contract_marks_canonical_rlm_as_repair_adapted(self) -> None:
        contract = RLMAdapter().contract()

        assert contract.repair_scope_policy == "single_file_only"
        assert (
            contract.prompt_contract_kind
            == "adapter_defined_tool_augmented_prompt_with_repair_fallback"
        )
        assert contract.output_contract == "single_file_repaired_source_or_final_answer_salvage"
        assert any("repair-specific final-turn fallback" in note for note in contract.notes)
        assert any("valid full-file final answer" in note for note in contract.notes)

    def test_contract_reflects_repair_fair_prompt_contract(self, tmp_path: Path) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        adapter.setup(
            BaselineConfig(
                repo_dir=repo,
                extra={"baseline_rlm_prompt_contract": "repair_fair_v1"},
            )
        )
        contract = adapter.contract()
        assert (
            contract.prompt_contract_kind
            == "adapter_defined_tool_augmented_prompt_with_repair_fallback__repair_fair_v1"
        )

    def test_run_repair_retries_and_accumulates_metrics(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=2,
            api_timeout_sec=11,
            process_timeout_sec=30,
            timeout_sec=None,
            extra={
                "baseline_rlm_max_depth": 2,
                "baseline_rlm_max_iterations": 6,
                "baseline_rlm_timeout_sec": 40,
                "baseline_rlm_max_errors": 4,
                "baseline_rlm_compaction": True,
                "baseline_rlm_compaction_threshold_pct": 0.8,
                "baseline_rlm_temperature": 0.2,
                "baseline_rlm_top_p": 0.9,
                "baseline_rlm_max_tokens": 512,
                "baseline_rlm_max_total_tokens": 4000,
            },
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        test_outcomes = iter(
            [
                _CommandOutcome(
                    returncode=1, stdout="", stderr="assert compute(3) == 6", duration_sec=0.1
                ),
                _CommandOutcome(returncode=1, stdout="", stderr="still wrong", duration_sec=0.2),
                _CommandOutcome(returncode=0, stdout="ok", stderr="", duration_sec=0.15),
            ]
        )

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            assert work_dir.is_dir()
            assert python_args == ["-m", "pytest", "-q"]
            assert timeout_sec == 15
            return next(test_outcomes)

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)

        responses = [
            {
                "response": "FINAL_VAR(final_code)",
                "execution_time": 0.4,
                "usage_summary": {
                    "total_calls": 3,
                    "total_input_tokens": 100,
                    "total_output_tokens": 40,
                },
                "char_summary": {
                    "total_calls": 3,
                    "total_input_chars": 900,
                    "total_output_chars": 120,
                },
                "call_summary": {
                    "call_finish_reason_counts": {"length": 1, "stop": 2},
                    "call_length_finish_count": 1,
                    "compat_recovery_count": 0,
                },
                "stop_reason": "final_answer",
                "error": "",
                "trajectory_summary": {
                    "iterations_completed": 4,
                    "subcall_count": 1,
                    "max_depth_reached": 2,
                    "tool_call_counts": {
                        "list_files": 1,
                        "read_file": 2,
                        "write_buggy_py": 1,
                        "run_case_tests": 1,
                    },
                    "test_run_count": 1,
                },
                "workspace_buggy_code": "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
                "workspace_buggy_changed": True,
            },
            {
                "response": "FINAL_VAR(final_code)",
                "execution_time": 0.3,
                "usage_summary": {
                    "total_calls": 2,
                    "total_input_tokens": 80,
                    "total_output_tokens": 20,
                },
                "char_summary": {
                    "total_calls": 2,
                    "total_input_chars": 700,
                    "total_output_chars": 90,
                },
                "call_summary": {
                    "call_finish_reason_counts": {"stop": 2},
                    "call_length_finish_count": 0,
                    "compat_recovery_count": 1,
                    "compat_recovery_events": [
                        {"reason": "request_validation_retry_disable_thinking"}
                    ],
                },
                "stop_reason": "final_answer",
                "error": "",
                "trajectory_summary": {
                    "iterations_completed": 3,
                    "subcall_count": 0,
                    "max_depth_reached": 0,
                    "tool_call_counts": {
                        "read_file": 1,
                        "write_buggy_py": 1,
                        "run_case_tests": 1,
                    },
                    "test_run_count": 1,
                },
                "workspace_buggy_code": "from helper import offset\n\n\ndef compute(value):\n    return offset(value)\n",
                "workspace_buggy_changed": True,
            },
        ]

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            assert request_payload["model"] == "test-model"
            assert request_payload["api_timeout_sec"] == 11.0
            assert request_payload["max_depth"] == 2
            assert request_payload["max_iterations"] == 6
            assert request_payload["max_timeout_sec"] == 30.0
            assert request_payload["max_errors"] == 4
            assert request_payload["compaction"] is True
            assert request_payload["temperature"] == 0.2
            assert request_payload["top_p"] == 0.9
            assert request_payload["max_tokens"] == 512
            assert request_payload["max_total_tokens"] == 4000
            assert "write_buggy_py(content)" in request_payload["root_prompt"]
            assert timeout == 30
            payload = responses.pop(0)
            workspace_dir = Path(request_payload["workspace_dir"])
            (workspace_dir / "buggy.py").write_text(
                payload["workspace_buggy_code"], encoding="utf-8"
            )
            response_path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.iterations_used == 2
        assert (
            result.repaired_code
            == "from helper import offset\n\n\ndef compute(value):\n    return offset(value)\n"
        )
        assert result.token_usage == {
            "prompt_tokens": 180,
            "completion_tokens": 60,
            "total_tokens": 240,
        }
        attempts = result.meta["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["candidate_status"] == "tests_still_failing"
        assert attempts[1]["candidate_status"] == "passed"
        assert attempts[1]["outcome_classification"] == "success_workspace_write"
        assert attempts[1]["compat_recovery_count"] == 1
        assert attempts[0]["call_finish_reason_counts"] == {"length": 1, "stop": 2}
        assert result.meta["terminal_outcome"] == "success_workspace_write"
        metrics = result.meta["metrics"]
        assert metrics["llm_call_count"] == 5
        assert metrics["reported_tokens_prompt"] == 180
        assert metrics["reported_tokens_completion"] == 60
        assert metrics["reported_chars_prompt"] == 1600
        assert metrics["reported_chars_completion"] == 210
        assert metrics["internal_iteration_count"] == 7
        assert metrics["internal_subcall_count"] == 1
        assert metrics["internal_test_run_count"] == 2
        assert metrics["internal_max_depth_reached"] == 2
        assert metrics["internal_length_finish_count"] == 1
        assert metrics["internal_guard_stop_count"] == 0
        assert metrics["internal_search_exhausted_attempt_count"] == 0
        assert metrics["internal_workspace_write_success_count"] == 1
        assert metrics["internal_salvage_success_count"] == 0
        assert metrics["internal_compat_recovery_count"] == 1
        assert metrics["direct_gen_passed"] is False
        assert metrics["debug_added_value"] is True

    def test_run_repair_includes_controller_guidance_in_root_prompt(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            timeout_sec=30,
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:4 in compute",
                    "top_facts": ["Return expression adds an unnecessary + 1 offset."],
                }
            },
            "history_bundle": {
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "Preserve offset(value) semantics and only remove the extra increment.",
                    }
                ]
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["must edit the crash file"],
            },
            "solver_hints": {
                "retry_plan": "Edit only the buggy.py return expression.",
                "strategy_summary": "Localize the arithmetic fix and keep the patch minimal.",
                "decomposition": [
                    {
                        "title": "localize_fix",
                        "question": "Remove the extra increment without widening the edit scope.",
                    }
                ],
            },
        }
        test_outcomes = iter(
            [
                _CommandOutcome(
                    returncode=1, stdout="", stderr="assert compute(3) == 6", duration_sec=0.1
                ),
                _CommandOutcome(returncode=0, stdout="ok", stderr="", duration_sec=0.1),
            ]
        )
        observed_prompt: dict[str, str] = {}

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            assert work_dir.is_dir()
            assert python_args == ["-m", "pytest", "-q"]
            assert timeout_sec == 15
            return next(test_outcomes)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            observed_prompt["root_prompt"] = str(request_payload["root_prompt"])
            workspace_dir = Path(request_payload["workspace_dir"])
            repaired = (
                "from helper import offset\n\n\ndef compute(value):\n    return offset(value)\n"
            )
            (workspace_dir / "buggy.py").write_text(repaired, encoding="utf-8")
            response_path.write_text(
                json.dumps(
                    {
                        "response": "FINAL_VAR(final_code)",
                        "execution_time": 0.2,
                        "usage_summary": {
                            "total_calls": 1,
                            "total_input_tokens": 22,
                            "total_output_tokens": 8,
                        },
                        "char_summary": {
                            "total_calls": 1,
                            "total_input_chars": 600,
                            "total_output_chars": 80,
                        },
                        "stop_reason": "final_answer",
                        "error": "",
                        "trajectory_summary": {
                            "iterations_completed": 2,
                            "subcall_count": 0,
                            "max_depth_reached": 0,
                            "tool_call_counts": {
                                "read_file": 1,
                                "write_buggy_py": 1,
                                "run_case_tests": 1,
                            },
                            "test_run_count": 1,
                        },
                        "workspace_buggy_code": repaired,
                        "workspace_buggy_changed": True,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)
        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        root_prompt = observed_prompt["root_prompt"]
        assert "controller_state: action=snapshot_lite outcome=materialized" in root_prompt
        assert "crash_anchor: buggy.py:4 in compute" in root_prompt
        assert "history_lessons: attempt 1: Preserve offset(value) semantics" in root_prompt
        assert "verifier_status: previous_verifier_rejected" in root_prompt
        assert "retry_plan: Edit only the buggy.py return expression." in root_prompt
        assert (
            "strategy_summary: Localize the arithmetic fix and keep the patch minimal."
            in root_prompt
        )
        assert (
            "decomposition: localize_fix: Remove the extra increment without widening the edit scope."
            in root_prompt
        )

    def test_build_root_prompt_freezes_prompt_contract(self) -> None:
        adapter = RLMAdapter()

        root_prompt = adapter._build_root_prompt(
            failure_feedback="AssertionError: expected 6, got 7",
            controller_guidance="crash_anchor: buggy.py:4 in compute",
            attempt_idx=2,
            failure_feedback_max_chars=3500,
        )

        assert "Repair buggy.py in the provided workspace so the case tests pass." in root_prompt
        assert "write_buggy_py(content)" in root_prompt
        assert "run_case_tests()" in root_prompt
        assert "Only modify buggy.py. Do not edit tests or helper files." in root_prompt
        assert "Prefer write_buggy_py(content) to persist your fix in the workspace." in root_prompt
        assert (
            "your final answer must be ONLY the full corrected contents of buggy.py" in root_prompt
        )
        assert "FINAL(...)" in root_prompt
        assert "FINAL_VAR(...)" in root_prompt
        assert "This is outer repair attempt 2. Do not repeat the previous failure." in root_prompt
        assert "Failure feedback from the previous attempt/test run:" in root_prompt
        assert "Adaptive controller guidance:" in root_prompt

    def test_build_root_prompt_truncates_failure_feedback_when_configured(self) -> None:
        adapter = RLMAdapter()

        root_prompt = adapter._build_root_prompt(
            failure_feedback="Z" * 300,
            controller_guidance="",
            attempt_idx=1,
            failure_feedback_max_chars=80,
        )

        assert "Failure feedback from the previous attempt/test run:" in root_prompt
        assert "Z" * 90 not in root_prompt
        assert "..." in root_prompt

    def test_build_root_prompt_adds_repair_fair_solve_first_instructions(self) -> None:
        adapter = RLMAdapter()

        root_prompt = adapter._build_root_prompt(
            failure_feedback="AssertionError: expected 6, got 7",
            controller_guidance="",
            attempt_idx=1,
            prompt_contract="repair_fair_v1",
            failure_feedback_max_chars=3500,
        )

        assert "Attempt a minimal direct repair before recursive subcalls." in root_prompt
        assert "Within the first two inner iterations" in root_prompt
        assert "Use llm_query() and rlm_query() only after the direct attempt" in root_prompt

    def test_run_repair_uses_configured_feedback_and_guidance_budgets(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            max_attempts=1,
            process_timeout_sec=30,
            extra={
                "baseline_rlm_controller_guidance_max_chars": 120,
                "baseline_rlm_failure_feedback_max_chars": 90,
                "baseline_rlm_failure_feedback_meta_max_chars": 70,
            },
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:4 in compute",
                    "top_facts": ["A" * 300, "B" * 300],
                },
            },
            "history_bundle": {"schema_version": "v1"},
            "verifier_report": {"schema_version": "v1"},
        }

        monkeypatch.setattr(
            adapter,
            "_run_case_tests",
            lambda **kwargs: _CommandOutcome(
                returncode=1, stdout="", stderr="Y" * 300, duration_sec=0.1
            ),
        )
        observed_prompt: dict[str, str] = {}

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            observed_prompt["root_prompt"] = str(request_payload["root_prompt"])
            workspace_dir = Path(request_payload["workspace_dir"])
            (workspace_dir / "buggy.py").write_text(
                "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
                encoding="utf-8",
            )
            response_path.write_text(
                json.dumps(
                    {
                        "response": "FINAL_VAR(final_code)",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "trajectory_summary": {},
                        "stop_reason": "workspace_unchanged",
                        "error": "",
                        "workspace_buggy_code": "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert "Adaptive controller guidance:" in observed_prompt["root_prompt"]
        assert len(observed_prompt["root_prompt"]) < 1200
        assert result.iterations_used == 1
        assert result.meta["attempts"][0]["candidate_status"] == "workspace_unchanged"

    def test_run_repair_uses_fair_v1_hybrid_guidance_block(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python-fair"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            max_attempts=1,
            process_timeout_sec=30,
            extra={
                "baseline_hybrid_guidance_mode": "fair_v1",
                "baseline_rlm_controller_guidance_max_chars": 360,
                "baseline_rlm_failure_feedback_max_chars": 90,
                "baseline_rlm_failure_feedback_meta_max_chars": 70,
            },
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:4 in compute",
                    "top_facts": ["Focus on buggy.py", "Do not widen the edit"],
                },
            },
            "history_bundle": {
                "schema_version": "v1",
                "lessons": [
                    {"source": "attempt_history", "attempt": 1, "lesson": "keep the return type"}
                ],
            },
            "verifier_report": {
                "schema_version": "v1",
                "status": "previous_verifier_rejected",
                "reasons": ["touch buggy.py", "preserve signature"],
            },
            "solver_hints": {
                "retry_plan": "Edit only the compute return expression.",
                "strategy_summary": "should not appear",
                "decomposition": [{"title": "localize", "question": "should not appear"}],
            },
        }

        monkeypatch.setattr(
            adapter,
            "_run_case_tests",
            lambda **kwargs: _CommandOutcome(
                returncode=1, stdout="", stderr="Y" * 200, duration_sec=0.1
            ),
        )
        observed_prompt: dict[str, str] = {}

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            observed_prompt["root_prompt"] = str(request_payload["root_prompt"])
            workspace_dir = Path(request_payload["workspace_dir"])
            (workspace_dir / "buggy.py").write_text(
                "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
                encoding="utf-8",
            )
            response_path.write_text(
                json.dumps(
                    {
                        "response": "FINAL_VAR(final_code)",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "trajectory_summary": {},
                        "stop_reason": "workspace_unchanged",
                        "error": "",
                        "workspace_buggy_code": "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        root_prompt = observed_prompt["root_prompt"]
        assert "Adaptive controller guidance:" in root_prompt
        assert "verifier: status=previous_verifier_rejected" in root_prompt
        assert "retry_plan: Edit only the compute return expression." in root_prompt
        assert "strategy_summary:" not in root_prompt
        assert "decomposition:" not in root_prompt

    def test_env_client_forwards_api_timeout_to_backend(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        captured: dict[str, object] = {}

        class _FakeLogger:
            def get_trajectory(self) -> dict[str, object]:
                return {"iterations": []}

        class _FakeCompletion:
            response = "patched"
            root_model = "test-model"
            metadata = {"iterations": []}

        class _FakeRLM:
            def __init__(
                self, *, backend: str, backend_kwargs: dict[str, object], **kwargs: object
            ) -> None:
                del backend, kwargs
                captured["backend_kwargs"] = dict(backend_kwargs)

            def completion(
                self, prompt_context: str, root_prompt: str | None = None
            ) -> _FakeCompletion:
                del prompt_context, root_prompt
                return _FakeCompletion()

        monkeypatch.setattr(
            env_client,
            "_install_tracking",
            lambda **kwargs: (_FakeRLM, _FakeLogger, []),
        )
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "api_timeout_sec": 7.5,
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        assert exit_code == 0
        assert captured["backend_kwargs"] == {
            "model_name": "test-model",
            "api_key": "sk-test",
            "base_url": None,
            "timeout": 7.5,
        }

    def test_env_client_patches_default_answer_to_request_buggy_py_contents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_default_answer_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        captured: dict[str, object] = {}
        repaired = "def compute(value):\n    return value * 2\n"

        class _FakeChatCompletions:
            def create(self, **kwargs: object) -> object:
                captured["messages"] = kwargs["messages"]
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content=repaired),
                            finish_reason="stop",
                        )
                    ]
                )

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeChatCompletions()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs: object) -> None:
                self.base_url = kwargs.get("base_url")
                self.chat = _FakeChat()

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = _FakeOpenAIClient
        openai_module.AsyncOpenAI = _FakeOpenAIClient
        monkeypatch.setitem(sys.modules, "openai", openai_module)
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["response"] == repaired.strip()
        assert response_payload["call_summary"]["call_finish_reason_counts"] == {"stop": 1}
        assert response_payload["call_summary"]["call_records"][0]["finish_reason"] == "stop"
        assert not response_payload["call_summary"]["call_records"][0][
            "request_validation_retry_applied"
        ]
        assert isinstance(captured["messages"], list)
        assert captured["messages"][-1]["role"] == "user"
        assert (
            "Return ONLY the full corrected contents of buggy.py"
            in (captured["messages"][-1]["content"])
        )

    def test_env_client_patches_child_default_answer_to_request_buggy_py_contents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_child_default_answer_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        captured_messages: list[list[dict[str, object]]] = []
        repaired = "def compute(value):\n    return value * 2\n"

        class _FakeChatCompletions:
            def create(self, **kwargs: object) -> object:
                messages = kwargs["messages"]
                assert isinstance(messages, list)
                captured_messages.append(messages)
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content=repaired),
                            finish_reason="stop",
                        )
                    ]
                )

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeChatCompletions()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs: object) -> None:
                self.base_url = kwargs.get("base_url")
                self.chat = _FakeChat()

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = _FakeOpenAIClient
        openai_module.AsyncOpenAI = _FakeOpenAIClient
        monkeypatch.setitem(sys.modules, "openai", openai_module)
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["response"] == repaired.strip()
        assert response_payload["call_summary"]["call_finish_reason_counts"] == {"stop": 1}
        assert captured_messages
        assert captured_messages[-1][0]["content"] == "child prompt"
        assert captured_messages[-1][-1]["role"] == "user"
        assert (
            "Return ONLY the full corrected contents of buggy.py"
            in (captured_messages[-1][-1]["content"])
        )

    def test_env_client_rejects_context_leakage_write_and_restores_workspace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        original_buggy = "print('buggy')\n"

        class _FakeLogger:
            def get_trajectory(self) -> dict[str, object]:
                return {"iterations": [{}]}

        class _FakeCompletion:
            response = ""
            root_model = "test-model"
            metadata = {"iterations": [{}]}

        class _FakeRLM:
            def __init__(self, *, custom_tools: dict[str, object], **kwargs: object) -> None:
                del kwargs
                self.custom_tools = custom_tools

            def completion(
                self, prompt_context: str, root_prompt: str | None = None
            ) -> _FakeCompletion:
                del prompt_context, root_prompt
                write_tool = self.custom_tools["write_buggy_py"]
                assert isinstance(write_tool, dict)
                write_result = write_tool["tool"]("print(context)\n")
                assert "Rejected write_buggy_py(content)" in write_result
                return _FakeCompletion()

        monkeypatch.setattr(
            env_client,
            "_install_tracking",
            lambda **kwargs: (_FakeRLM, _FakeLogger, []),
        )
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text(original_buggy, encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["workspace_buggy_code"] == original_buggy
        assert response_payload["workspace_buggy_changed"] is False
        assert response_payload["trajectory_summary"]["write_rejection_count"] == 1
        assert response_payload["trajectory_summary"]["last_write_rejection"]["reason"] == (
            "context_reference_leakage"
        )
        assert (
            "print(context)"
            in response_payload["trajectory_summary"]["last_write_rejection"]["detail"]
        )

    def test_env_client_uses_workspace_override_after_internal_tests_pass(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        repaired = "VALUE = 1\n"

        class _FakeLogger:
            def get_trajectory(self) -> dict[str, object]:
                return {"iterations": [{"final_answer": "done"}]}

        class _FakeCompletion:
            def __init__(self, response: str) -> None:
                self.response = response
                self.root_model = "test-model"
                self.metadata = {"iterations": [{"final_answer": "done"}]}

        class _ExplodingHandler:
            def completion(self, message_history: list[dict[str, object]]) -> str:
                del message_history
                raise AssertionError("lm_handler.completion should not run after tests pass")

        class _FakeRLM:
            def __init__(self, *, custom_tools: dict[str, object], **kwargs: object) -> None:
                del kwargs
                self.custom_tools = custom_tools

            def completion(
                self, prompt_context: str, root_prompt: str | None = None
            ) -> _FakeCompletion:
                del prompt_context, root_prompt
                write_tool = self.custom_tools["write_buggy_py"]
                test_tool = self.custom_tools["run_case_tests"]
                assert isinstance(write_tool, dict)
                assert isinstance(test_tool, dict)
                write_result = write_tool["tool"](repaired)
                assert "Wrote" in write_result
                test_result = test_tool["tool"]()
                assert "Exit code: 0" in test_result
                response = self._default_answer(
                    [{"role": "user", "content": "repair prompt"}],
                    _ExplodingHandler(),
                )
                return _FakeCompletion(response)

        monkeypatch.setattr(
            env_client,
            "_install_tracking",
            lambda **kwargs: (_FakeRLM, _FakeLogger, []),
        )
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                    "test_python": sys.executable,
                    "python_args": ["-c", "import buggy"],
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("VALUE = 0\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["response"] == repaired
        assert response_payload["workspace_buggy_code"] == repaired
        assert response_payload["workspace_buggy_changed"] is True
        assert response_payload["trajectory_summary"]["workspace_tests_passed"] is True
        assert response_payload["trajectory_summary"]["test_run_count"] == 1

    def test_run_repair_salvages_valid_final_answer_when_workspace_is_unchanged(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            process_timeout_sec=30,
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        repaired = "from helper import offset\n\n\ndef compute(value):\n    return offset(value)"

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            del python_args, timeout_sec
            candidate = (work_dir / "buggy.py").read_text(encoding="utf-8")
            if candidate == repaired:
                return _CommandOutcome(returncode=0, stdout="passed", stderr="", duration_sec=0.1)
            return _CommandOutcome(returncode=1, stdout="", stderr="failing", duration_sec=0.1)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            response_path = Path(cmd[3])
            response_path.write_text(
                json.dumps(
                    {
                        "response": repaired,
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "call_summary": {
                            "call_finish_reason_counts": {"length": 1},
                            "call_length_finish_count": 1,
                            "compat_recovery_count": 0,
                        },
                        "trajectory_summary": {
                            "iterations_completed": 12,
                            "subcall_count": 0,
                            "max_depth_reached": 0,
                            "tool_call_counts": {
                                "list_files": 1,
                                "read_file": 1,
                            },
                            "test_run_count": 0,
                        },
                        "stop_reason": "iteration_budget_exhausted",
                        "error": "",
                        "workspace_buggy_code": (rlm_case / "buggy.py").read_text(encoding="utf-8"),
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)
        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.repaired_code == repaired
        assert result.meta["attempts"][0]["candidate_source"] == "final_answer_salvage"
        assert result.meta["attempts"][0]["candidate_salvaged"] is True
        assert (
            result.meta["attempts"][0]["outcome_classification"] == "success_final_answer_salvage"
        )
        assert result.meta["attempts"][0]["call_length_finish_count"] == 1
        assert result.meta["attempts"][0]["stop_due_to_guard"] is True
        assert result.meta["attempts"][0]["guard_name"] == "iteration_budget_exhausted"
        assert result.meta["attempts"][0]["workspace_buggy_changed"] is False
        assert result.meta["attempts"][0]["write_buggy_py_calls"] == 0
        assert result.meta["attempts"][0]["no_write_attempt_streak"] == 1
        assert result.meta["terminal_outcome"] == "success_final_answer_salvage"
        assert result.meta["metrics"]["internal_guard_stop_count"] == 1
        assert result.meta["metrics"]["internal_length_finish_count"] == 1
        assert result.meta["metrics"]["internal_salvage_success_count"] == 1

    def test_run_repair_salvages_wrapped_fenced_answer_when_workspace_is_unchanged(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            process_timeout_sec=30,
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        repaired = "from helper import offset\n\n\ndef compute(value):\n    return offset(value)"

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            del python_args, timeout_sec
            candidate = (work_dir / "buggy.py").read_text(encoding="utf-8")
            if candidate == repaired:
                return _CommandOutcome(returncode=0, stdout="passed", stderr="", duration_sec=0.1)
            return _CommandOutcome(returncode=1, stdout="", stderr="failing", duration_sec=0.1)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            response_path = Path(cmd[3])
            response_path.write_text(
                json.dumps(
                    {
                        "response": f"Here is the fix:\n```python\n{repaired}\n```",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "call_summary": {
                            "call_finish_reason_counts": {"stop": 1},
                            "call_length_finish_count": 0,
                            "compat_recovery_count": 0,
                        },
                        "trajectory_summary": {
                            "iterations_completed": 12,
                            "subcall_count": 0,
                            "max_depth_reached": 0,
                            "tool_call_counts": {
                                "list_files": 1,
                                "read_file": 1,
                            },
                            "test_run_count": 0,
                        },
                        "stop_reason": "iteration_budget_exhausted",
                        "error": "",
                        "workspace_buggy_code": (rlm_case / "buggy.py").read_text(encoding="utf-8"),
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)
        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.repaired_code == repaired
        assert result.meta["attempts"][0]["candidate_source"] == "final_answer_salvage"
        assert result.meta["attempts"][0]["candidate_salvaged"] is True
        assert (
            result.meta["attempts"][0]["outcome_classification"] == "success_final_answer_salvage"
        )
        assert result.meta["terminal_outcome"] == "success_final_answer_salvage"

    def test_salvage_candidate_accepts_expression_only_module(self) -> None:
        adapter = RLMAdapter()

        salvaged, salvage_status = adapter._salvage_candidate_from_response(
            raw_response="print([*input()])",
            original_code="print(input())\n",
        )

        assert salvaged == "print([*input()])"
        assert salvage_status == "final_answer_salvage"

    def test_salvage_candidate_extracts_wrapped_fenced_python(self) -> None:
        adapter = RLMAdapter()
        repaired = "from helper import offset\n\n\ndef compute(value):\n    return offset(value)\n"

        salvaged, salvage_status = adapter._salvage_candidate_from_response(
            raw_response=f"Here is the fix:\n```python\n{repaired}```",
            original_code="def compute(value):\n    return value + 1\n",
        )

        assert salvaged == repaired.strip()
        assert salvage_status == "final_answer_salvage"

    def test_salvage_candidate_extracts_later_line_final_answer(self) -> None:
        adapter = RLMAdapter()
        repaired = "from helper import offset\n\n\ndef compute(value):\n    return offset(value)"

        salvaged, salvage_status = adapter._salvage_candidate_from_response(
            raw_response=f"Repair complete.\nFINAL({repaired})\nExtra note: done.",
            original_code="def compute(value):\n    return value + 1\n",
        )

        assert salvaged == repaired
        assert salvage_status == "final_answer_salvage"

    @pytest.mark.parametrize(
        "raw_response", ["FINAL(done)", 'FINAL("def compute(x):\\n    return x")']
    )
    def test_salvage_candidate_rejects_scalar_final_answer_payloads(
        self,
        raw_response: str,
    ) -> None:
        adapter = RLMAdapter()

        salvaged, salvage_status = adapter._salvage_candidate_from_response(
            raw_response=raw_response,
            original_code="print(input())\n",
        )

        assert salvaged is None
        assert salvage_status == "response_not_salvageable"

    def test_salvage_candidate_rejects_unresolved_final_var_wrapper(self) -> None:
        adapter = RLMAdapter()

        salvaged, salvage_status = adapter._salvage_candidate_from_response(
            raw_response="FINAL_VAR(final_code)",
            original_code="print(input())\n",
        )

        assert salvaged is None
        assert salvage_status == "response_not_salvageable"

    def test_run_repair_escalates_feedback_after_read_only_attempt(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=2,
            process_timeout_sec=30,
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        observed_prompts: list[str] = []

        monkeypatch.setattr(
            adapter,
            "_run_case_tests",
            lambda **kwargs: _CommandOutcome(
                returncode=1, stdout="", stderr="failing", duration_sec=0.1
            ),
        )

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            observed_prompts.append(str(request_payload["root_prompt"]))
            response_path.write_text(
                json.dumps(
                    {
                        "response": "",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "trajectory_summary": {
                            "iterations_completed": 12,
                            "subcall_count": 0,
                            "max_depth_reached": 0,
                            "tool_call_counts": {
                                "list_files": 1,
                                "read_file": 2,
                            },
                            "test_run_count": 0,
                        },
                        "stop_reason": "iteration_budget_exhausted",
                        "error": "",
                        "workspace_buggy_code": (rlm_case / "buggy.py").read_text(encoding="utf-8"),
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert len(observed_prompts) == 2
        assert "This attempt did not call write_buggy_py(content)." in observed_prompts[1]
        assert result.meta["attempts"][0]["stop_due_to_guard"] is True
        assert result.meta["attempts"][0]["guard_name"] == "iteration_budget_exhausted"
        assert result.meta["attempts"][0]["outcome_classification"] == "failure_search_exhausted"
        assert result.meta["terminal_outcome"] == "failure_search_exhausted"
        assert result.meta["metrics"]["internal_guard_stop_count"] == 2
        assert result.meta["metrics"]["internal_search_exhausted_attempt_count"] == 2

    def test_run_repair_marks_invalid_workspace_write_from_env_client(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            process_timeout_sec=30,
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()

        precheck_calls = {"count": 0}

        def _precheck_only(**kwargs: object) -> _CommandOutcome:
            del kwargs
            precheck_calls["count"] += 1
            return _CommandOutcome(returncode=1, stdout="", stderr="failing", duration_sec=0.1)

        monkeypatch.setattr(adapter, "_run_case_tests", _precheck_only)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            response_path = Path(cmd[3])
            response_path.write_text(
                json.dumps(
                    {
                        "response": "",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "call_summary": {},
                        "trajectory_summary": {
                            "iterations_completed": 12,
                            "subcall_count": 0,
                            "max_depth_reached": 0,
                            "tool_call_counts": {
                                "write_buggy_py": 1,
                                "read_file": 1,
                            },
                            "test_run_count": 0,
                            "write_rejection_count": 1,
                            "last_write_rejection": {
                                "reason": "context_reference_leakage",
                                "detail": "print(context)",
                            },
                        },
                        "stop_reason": "completed",
                        "error": "",
                        "workspace_buggy_code": (rlm_case / "buggy.py").read_text(encoding="utf-8"),
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.repaired_code is None
        assert result.meta["error"] == "invalid_workspace_write"
        assert precheck_calls["count"] == 1
        assert result.meta["attempts"][0]["candidate_status"] == "invalid_workspace_write"
        assert (
            result.meta["attempts"][0]["outcome_classification"]
            == "failure_invalid_workspace_write"
        )
        assert result.meta["attempts"][0]["write_rejection_count"] == 1
        assert result.meta["attempts"][0]["last_write_rejection"]["reason"] == (
            "context_reference_leakage"
        )
        assert result.meta["terminal_outcome"] == "failure_invalid_workspace_write"

    @pytest.mark.parametrize(
        ("base_url", "expected_base_url"),
        [
            ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
            ("http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1"),
            ("https://api.z.ai/api/anthropic", "https://api.z.ai/api/anthropic"),
        ],
    )
    def test_env_client_normalizes_openai_base_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        base_url: str,
        expected_base_url: str,
    ) -> None:
        repo = _make_env_client_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        captured: dict[str, object] = {}

        class _FakeLogger:
            def get_trajectory(self) -> dict[str, object]:
                return {"iterations": []}

        class _FakeCompletion:
            response = "patched"
            root_model = "test-model"
            metadata = {"iterations": []}

        class _FakeRLM:
            def __init__(
                self, *, backend: str, backend_kwargs: dict[str, object], **kwargs: object
            ) -> None:
                del backend, kwargs
                captured["backend_kwargs"] = dict(backend_kwargs)

            def completion(
                self, prompt_context: str, root_prompt: str | None = None
            ) -> _FakeCompletion:
                del prompt_context, root_prompt
                return _FakeCompletion()

        monkeypatch.setattr(
            env_client,
            "_install_tracking",
            lambda **kwargs: (_FakeRLM, _FakeLogger, []),
        )
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "base_url": base_url,
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        assert exit_code == 0
        assert captured["backend_kwargs"] == {
            "model_name": "test-model",
            "api_key": "sk-test",
            "base_url": expected_base_url,
        }

    def test_run_repair_returns_compatibility_error_from_env_client(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_rlm_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            api_timeout_sec=11,
            process_timeout_sec=30,
            extra={"baseline_openai_compat_profile": "lmstudio_model_yaml"},
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            del work_dir, python_args, timeout_sec
            return _CommandOutcome(returncode=1, stdout="", stderr="failing", duration_sec=0.1)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            response_path = Path(cmd[3])
            response_path.write_text(
                json.dumps(
                    {
                        "error": "openai_compat_reasoning_only",
                        "compat_message": "Disable thinking in the local model template.",
                        "compat_meta": {
                            "compat_profile_applied": "lmstudio_model_yaml",
                            "empty_content_with_reasoning": True,
                        },
                        "response": "",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "trajectory_summary": {},
                        "stop_reason": "openai_compat_reasoning_only",
                        "workspace_buggy_code": "",
                        "workspace_buggy_changed": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)
        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.iterations_used == 1
        assert result.meta["error"] == "openai_compat_reasoning_only"
        assert result.meta["compat_message"] == "Disable thinking in the local model template."
        assert result.meta["compat_meta"]["compat_profile_applied"] == "lmstudio_model_yaml"
        assert result.meta["terminal_outcome"] == "openai_compat_reasoning_only"

    def test_run_repair_surfaces_request_validation_compat_error(
        self,
        rlm_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_tracking_repo(tmp_path)
        adapter = RLMAdapter()
        fake_python = tmp_path / "rlm-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            api_timeout_sec=11,
            process_timeout_sec=30,
        )
        adapter.setup(config)
        case_input = convert_case(
            rlm_case,
            case_context=BaselineCaseContext(
                case_id="rlm_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()

        class _BadRequestError(Exception):
            pass

        class _FailingChatCompletions:
            def create(self, **kwargs: object) -> object:
                del kwargs
                raise _BadRequestError(
                    "Error code: 400 - {'error': {'code': 400, 'message': "
                    "'Assistant response prefill is incompatible with "
                    "enable_thinking.', 'type': 'invalid_request_error'}}"
                )

        class _FailingChat:
            def __init__(self) -> None:
                self.completions = _FailingChatCompletions()

        class _FailingOpenAIClient:
            def __init__(self, **kwargs: object) -> None:
                self.base_url = kwargs.get("base_url")
                self.chat = _FailingChat()

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = _FailingOpenAIClient
        openai_module.AsyncOpenAI = _FailingOpenAIClient
        monkeypatch.setitem(sys.modules, "openai", openai_module)

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            del work_dir, python_args, timeout_sec
            return _CommandOutcome(returncode=1, stdout="", stderr="failing", duration_sec=0.1)

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            for module_name in list(sys.modules):
                if module_name == "rlm" or module_name.startswith("rlm."):
                    monkeypatch.delitem(sys.modules, module_name, raising=False)
            exit_code = env_client.main([cmd[1], cmd[2], cmd[3]])
            return subprocess.CompletedProcess(cmd, exit_code, stdout="", stderr="")

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)
        monkeypatch.setattr("baselines.rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.meta["error"] == "openai_compat_reasoning_only"
        assert result.meta["compat_meta"]["request_validation_error"] is True
        assert result.meta["compat_meta"]["request_validation_failure_kind"] == (
            "assistant_prefill_enable_thinking_conflict"
        )
        assert result.meta["compat_meta"]["compat_profile_applied"] == "qwen_openai_raw"
        assert "enable_thinking" in result.meta["compat_message"]
        assert "disable thinking in the server or model template" in result.meta["compat_message"]
        assert result.meta["terminal_outcome"] == "openai_compat_reasoning_only"
        assert not result.meta["attempts"]

    def test_env_client_auto_recovers_prefill_enable_thinking_conflict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_tracking_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        repaired = "def compute(value):\n    return value * 2\n"
        seen_requests: list[dict[str, object]] = []

        class _BadRequestError(Exception):
            pass

        class _FakeChatCompletions:
            def create(self, **kwargs: object) -> object:
                seen_requests.append(dict(kwargs))
                extra_body = kwargs.get("extra_body")
                chat_template_kwargs = (
                    extra_body.get("chat_template_kwargs") if isinstance(extra_body, dict) else None
                )
                if chat_template_kwargs != {"enable_thinking": False}:
                    raise _BadRequestError(
                        "Error code: 400 - {'error': {'code': 400, 'message': "
                        "'Assistant response prefill is incompatible with "
                        "enable_thinking.', 'type': 'invalid_request_error'}}"
                    )
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content=repaired),
                            finish_reason="stop",
                        )
                    ]
                )

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeChatCompletions()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs: object) -> None:
                self.base_url = kwargs.get("base_url")
                self.chat = _FakeChat()

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = _FakeOpenAIClient
        openai_module.AsyncOpenAI = _FakeOpenAIClient
        monkeypatch.setitem(sys.modules, "openai", openai_module)
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["error"] == ""
        assert response_payload["response"] == repaired.strip()
        assert response_payload["call_summary"]["compat_recovery_count"] == 1
        assert response_payload["call_summary"]["call_finish_reason_counts"] == {"stop": 1}
        assert response_payload["call_summary"]["call_records"][0]["finish_reason"] == "stop"
        assert response_payload["call_summary"]["call_records"][0][
            "request_validation_retry_applied"
        ]
        assert len(seen_requests) == 2
        assert "extra_body" not in seen_requests[0]
        assert seen_requests[1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_env_client_preserves_original_compat_error_when_retry_fails_differently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_env_client_tracking_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        seen_requests: list[dict[str, object]] = []

        class _BadRequestError(Exception):
            pass

        class _FakeChatCompletions:
            def create(self, **kwargs: object) -> object:
                seen_requests.append(dict(kwargs))
                extra_body = kwargs.get("extra_body")
                chat_template_kwargs = (
                    extra_body.get("chat_template_kwargs") if isinstance(extra_body, dict) else None
                )
                if chat_template_kwargs is None:
                    raise _BadRequestError(
                        "Error code: 400 - {'error': {'code': 400, 'message': "
                        "'Assistant response prefill is incompatible with "
                        "enable_thinking.', 'type': 'invalid_request_error'}}"
                    )
                raise RuntimeError("unknown field chat_template_kwargs")

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeChatCompletions()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs: object) -> None:
                self.base_url = kwargs.get("base_url")
                self.chat = _FakeChat()

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = _FakeOpenAIClient
        openai_module.AsyncOpenAI = _FakeOpenAIClient
        monkeypatch.setitem(sys.modules, "openai", openai_module)
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "workspace_dir": str(tmp_path),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "prompt_context": "ctx",
                    "root_prompt": "root",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["error"] == "openai_compat_reasoning_only"
        assert response_payload["stop_reason"] == "openai_compat_reasoning_only"
        assert response_payload["compat_meta"]["request_validation_error"] is True
        assert response_payload["compat_meta"]["request_validation_failure_kind"] == (
            "assistant_prefill_enable_thinking_conflict"
        )
        assert response_payload["compat_meta"]["compat_profile_applied"] == "auto"
        assert "qwen_openai_raw" in response_payload["compat_message"]
        assert (
            "disable thinking in the server or model template" in response_payload["compat_message"]
        )
        assert len(seen_requests) == 2
        assert "extra_body" not in seen_requests[0]
        assert seen_requests[1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
