"""Tests for the INTERVENOR baseline adapter and bridge converter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from baselines.adapter_base import BaselineCaseContext, BaselineConfig, BaselineResult
from baselines.controller_packet import parse_controller_packet
from baselines.intervenor import env_client
from baselines.intervenor.adapter import IntervenorAdapter
from baselines.intervenor.case_converter import _build_prompt, _extract_entry_point, convert_case
from baselines.openai_compat import OpenAICompatResponseError


def _extract_diff(adapter: Any, case_dir: Path, repaired_code: str) -> str | None:
    """BAS-18: the adapters diff repaired code via ``BaselineAdapter.extract_diff``."""
    return adapter.extract_diff(case_dir, BaselineResult(repaired_code=repaired_code, success=True))


@pytest.fixture()
def simple_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "my_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        "def multiply(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text("CONSTANT = 12\n", encoding="utf-8")
    return case_dir


def _make_intervenor_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_intervenor"
    prompt_dir = repo / "prompt"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "code_teacher.txt").write_text(
        "Teacher prompt\n- buggy code:\n%%%buggy_code%%%\nError: %%%error_message%%%\n"
        "Modification method:\n",
        encoding="utf-8",
    )
    (prompt_dir / "code_learner.txt").write_text(
        "Learner prompt\n- buggy code:\n%%%buggy_code%%%\nError: %%%error_message%%%\n"
        "- modification method:\n%%%repair_method%%%\nCorrect the code:\n",
        encoding="utf-8",
    )
    return repo


class TestCaseConverter:
    def test_convert_case_packages_workspace_and_context(self, simple_case: Path) -> None:
        context = BaselineCaseContext(
            case_id="my_case",
            python_args=("-m", "pytest", "-q", "test_buggy.py"),
            timeout_sec=11,
        )

        result = convert_case(simple_case, case_context=context)

        assert result.task_id == "my_case"
        assert result.entry_point == "multiply"
        assert result.case_information == {
            "case_info_policy": "neutral_v1",
            "test_source_exposure": "full",
            "prompt_files": ["buggy.py", "test_buggy.py"],
        }
        assert result.python_args == ["-m", "pytest", "-q", "test_buggy.py"]
        assert result.timeout_sec == 11
        assert result.buggy_code == (simple_case / "buggy.py").read_text(encoding="utf-8")
        paths = {row["path"] for row in result.files}
        assert paths == {"buggy.py", "helper.py", "test_buggy.py"}

    def test_convert_case_prefers_case_context_case_id(self, simple_case: Path) -> None:
        adapter = IntervenorAdapter()

        result = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="canonical_case_id",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=11,
            ),
        )

        assert result.task_id == "canonical_case_id"

    def test_extract_entry_point_prefers_buggy_definition(self) -> None:
        assert _extract_entry_point("def compute(x):\n    return x\n", "") == "compute"

    def test_extract_entry_point_falls_back_to_import(self) -> None:
        assert _extract_entry_point("x = 1\n", "from buggy import transform\n") == "transform"

    def test_build_prompt_uses_signature_and_full_test_source(self) -> None:
        test_code = "def test_multiply():\n    assert multiply(2, 3) == 6\n"
        prompt = _build_prompt(
            'def multiply(a, b):\n    """Multiply values."""\n    return a * b\n',
            test_code,
            "multiply",
        )
        assert "Primary symbol: multiply" in prompt
        assert "Definition:" in prompt
        assert "Reference tests (test_buggy.py):\n" + test_code in prompt
        assert "Description:" not in prompt

    def test_render_learner_prompt_appends_output_contract_by_default(self, tmp_path: Path) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))

        prompt = adapter._render_learner_prompt(
            buggy_code="def multiply(a, b):\n    return a + b\n",
            error_message="AssertionError",
            repair_method="Use * instead of +",
            task_context="Primary symbol: multiply",
            controller_guidance="crash_anchor: buggy.py:2 in multiply",
        )

        assert "Task context:\nPrimary symbol: multiply" in prompt
        assert "Adaptive controller guidance:\ncrash_anchor: buggy.py:2 in multiply" in prompt
        assert "YOUR TASK: Output the full corrected Python source of buggy.py." in prompt

    def test_render_learner_prompt_can_skip_output_contract_suffix(self, tmp_path: Path) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))

        prompt = adapter._render_learner_prompt(
            buggy_code="def multiply(a, b):\n    return a + b\n",
            error_message="AssertionError",
            repair_method="Use * instead of +",
            task_context="Primary symbol: multiply",
            append_output_contract=False,
        )

        assert "YOUR TASK: Output the full corrected Python source of buggy.py." not in prompt

    def test_render_controller_guidance_respects_explicit_max_chars(self, tmp_path: Path) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))
        packet = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_decision_reason": "policy_action_snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": ["A" * 200, "B" * 200, "C" * 200],
                },
            },
            "history_bundle": {"schema_version": "v1"},
            "verifier_report": {"schema_version": "v1"},
        }

        compact = adapter._render_controller_guidance(
            controller_packet=parse_controller_packet(packet),
            focus="teacher",
            max_chars=120,
        )

        assert compact
        assert len(compact) <= 120


class TestResultExtractor:
    def test_make_diff_produces_valid_diff(self, simple_case: Path) -> None:
        diff = _extract_diff(
            IntervenorAdapter(), simple_case, "def multiply(a, b):\n    return a * b\n"
        )
        assert diff is not None
        assert "-    return a + b" in diff
        assert "+    return a * b" in diff

    def test_make_diff_identical_returns_none(self, simple_case: Path) -> None:
        same = (simple_case / "buggy.py").read_text(encoding="utf-8")
        assert _extract_diff(IntervenorAdapter(), simple_case, same) is None


class TestIntervenorAdapter:
    @staticmethod
    def _assert_attempt_metric_shape(metrics: dict[str, object]) -> None:
        assert "teacher_duration_sec_total" in metrics
        assert "learner_duration_sec_total" in metrics
        assert "postcheck_duration_sec_total" in metrics
        assert "learner_empty_response_count" in metrics
        assert "attempts_with_postcheck_count" in metrics

    def test_adapter_name(self) -> None:
        assert IntervenorAdapter().name == "intervenor"

    def test_setup_raises_on_missing_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            IntervenorAdapter().setup(BaselineConfig(repo_dir=tmp_path / "missing"))

    def test_setup_requires_prompt_assets(self, tmp_path: Path) -> None:
        repo = tmp_path / "fake_intervenor"
        repo.mkdir()
        with pytest.raises(RuntimeError, match="missing required prompt asset"):
            IntervenorAdapter().setup(BaselineConfig(repo_dir=repo))

    def test_setup_loads_prompt_assets(self, tmp_path: Path) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))
        assert "Modification method" in adapter._teacher_prompt_template
        assert "Correct the code" in adapter._learner_prompt_template

    def test_convert_case_delegates(self, simple_case: Path) -> None:
        adapter = IntervenorAdapter()
        result = adapter.convert_case(simple_case)
        assert result.task_id == "my_case"
        assert result.entry_point == "multiply"

    def test_run_repair_uses_repo_prompts_and_local_tests(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
            timeout_sec=20,
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        prompts: list[str] = []

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            prompts.append(prompt)
            if "Modification method:" in prompt:
                return "Change the operator from + to *.", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 7,
                "completion_tokens": 8,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.repaired_code == "def multiply(a, b):\n    return a * b"
        assert result.iterations_used == 1
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 13,
            "total_tokens": 30,
        }
        metrics = result.meta["metrics"]
        assert metrics["llm_call_count"] == 2
        assert metrics["llm_duration_sec"] == 0.0
        assert metrics["eval_duration_sec"] >= 0.0
        assert metrics["internal_total_duration_sec"] >= metrics["eval_duration_sec"]
        assert metrics["usage_available"] is True
        assert metrics["usage_reported_call_count"] == 2
        assert metrics["usage_missing_call_count"] == 0
        assert metrics["reported_tokens_prompt"] == 17
        assert metrics["reported_tokens_completion"] == 13
        assert metrics["reported_tokens_total"] == 30
        assert metrics["prompt_chars_total"] == sum(len(prompt) for prompt in prompts)
        assert metrics["completion_chars_total"] == (
            len("Change the operator from + to *.") + len("def multiply(a, b):\n    return a * b\n")
        )
        assert metrics["chars_total"] == (
            metrics["prompt_chars_total"] + metrics["completion_chars_total"]
        )
        assert metrics["reported_chars_prompt"] == metrics["prompt_chars_total"]
        assert metrics["reported_chars_completion"] == metrics["completion_chars_total"]
        assert metrics["reported_chars_total"] == metrics["chars_total"]
        assert metrics["usage_source"] == "provider_reported"
        self._assert_attempt_metric_shape(metrics)
        assert metrics["teacher_duration_sec_total"] == 0.0
        assert metrics["learner_duration_sec_total"] == 0.0
        assert metrics["postcheck_duration_sec_total"] >= 0.0
        assert metrics["learner_empty_response_count"] == 0
        assert metrics["attempts_with_postcheck_count"] == 1
        assert any("Teacher prompt" in prompt for prompt in prompts)
        assert any("Learner prompt" in prompt for prompt in prompts)
        assert any("Task context:" in prompt for prompt in prompts)

    def test_run_repair_respects_configured_token_and_output_contract_knobs(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
            extra={
                "baseline_intervenor_teacher_max_tokens": 111,
                "baseline_intervenor_learner_max_tokens": 222,
                "baseline_intervenor_append_output_contract": False,
            },
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        observed: list[tuple[str, int]] = []

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            observed.append((prompt, max_tokens))
            if "Modification method:" in prompt:
                return "Change the operator from + to *.", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 7,
                "completion_tokens": 8,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert observed[0][1] == 111
        assert observed[1][1] == 222
        assert (
            "YOUR TASK: Output the full corrected Python source of buggy.py." not in observed[1][0]
        )

    def test_run_repair_records_optional_teacher_and_learner_finish_reasons(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(repo_dir=repo, model="test-model", max_attempts=1)
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            del client, model, max_tokens, config
            if "Modification method:" in prompt:
                return (
                    "Change the operator from + to *.",
                    {"prompt_tokens": 5, "completion_tokens": 3},
                    0.01,
                    {"finish_reason": "stop"},
                )
            return (
                "def multiply(a, b):\n    return a * b\n",
                {"prompt_tokens": 7, "completion_tokens": 8},
                0.02,
                {"finish_reason": "length"},
            )

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        attempts = result.meta["attempts"]
        assert attempts[0]["teacher_finish_reason"] == "stop"
        assert attempts[0]["teacher_response_saturated"] is False
        assert attempts[0]["learner_finish_reason"] == "length"
        assert attempts[0]["learner_response_saturated"] is True
        metrics = result.meta["metrics"]
        assert metrics["teacher_finish_reason_known_count"] == 1
        assert metrics["learner_finish_reason_known_count"] == 1
        assert metrics["teacher_finish_reason_length_count"] == 0
        assert metrics["learner_finish_reason_length_count"] == 1

    def test_generate_text_direct_path_requests_usage_and_parses_input_output_tokens(
        self,
        tmp_path: Path,
    ) -> None:
        adapter = IntervenorAdapter()
        observed: dict[str, object] = {}

        class _Usage:
            input_tokens = 11
            output_tokens = 5
            total_tokens = 16

        class _Response:
            def __init__(self) -> None:
                self.choices = [
                    SimpleNamespace(message=SimpleNamespace(content="patched")),
                ]
                self.usage = _Usage()

        class _ChatCompletions:
            def create(self, **kwargs: object) -> _Response:
                observed["kwargs"] = dict(kwargs)
                return _Response()

        client = SimpleNamespace(chat=SimpleNamespace(completions=_ChatCompletions()))
        config = BaselineConfig(
            repo_dir=tmp_path,
            model="test-model",
            api_base_url="http://127.0.0.1:9999",
            api_key="sk-test",
        )

        text, usage, duration_sec, meta = adapter._generate_text(
            client=client,
            model="test-model",
            prompt="hello",
            max_tokens=32,
            config=config,
        )

        assert text == "patched"
        assert usage == {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }
        assert duration_sec >= 0.0
        assert meta == {}
        assert observed["kwargs"]["extra_body"] == {}  # BAS-17: only for OpenRouter

    def test_env_client_requests_usage_and_maps_input_output_tokens(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "model": "test-model",
                    "prompt": "hello",
                    "max_tokens": 32,
                    "base_url": "http://127.0.0.1:9999",
                    "api_key": "sk-test",
                    "timeout_sec": 7,
                }
            ),
            encoding="utf-8",
        )
        observed: dict[str, object] = {}

        class _Usage:
            input_tokens = 11
            output_tokens = 5
            total_tokens = 16

        class _Response:
            def __init__(self) -> None:
                self.choices = [
                    SimpleNamespace(message=SimpleNamespace(content="patched")),
                ]
                self.usage = _Usage()

        class _ChatCompletions:
            def create(self, **kwargs: object) -> _Response:
                observed["kwargs"] = dict(kwargs)
                return _Response()

        class _OpenAI:
            def __init__(
                self,
                *,
                base_url: str | None = None,
                api_key: str | None = None,
                timeout: float | None = None,
                max_retries: int | None = None,
            ) -> None:
                observed["constructor"] = {
                    "base_url": base_url,
                    "api_key": api_key,
                    "timeout": timeout,
                    "max_retries": max_retries,
                }
                self.chat = SimpleNamespace(completions=_ChatCompletions())

        monkeypatch.setattr(
            env_client, "import_module", lambda name: SimpleNamespace(OpenAI=_OpenAI)
        )

        exit_code = env_client.main(
            [
                "env_client.py",
                str(request_path),
                str(response_path),
            ]
        )

        assert exit_code == 0
        assert observed["constructor"] == {
            "base_url": "http://127.0.0.1:9999/v1",
            "api_key": "sk-test",
            "timeout": 7.0,
            "max_retries": 0,
        }
        assert observed["kwargs"]["extra_body"] == {}  # BAS-17: only for OpenRouter
        assert json.loads(response_path.read_text(encoding="utf-8")) == {
            "text": "patched",
            "meta": {},
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
            },
        }

    def test_generate_text_raises_on_reasoning_only_response(self, tmp_path: Path) -> None:
        adapter = IntervenorAdapter()

        class _Response:
            def __init__(self) -> None:
                self.choices = [
                    SimpleNamespace(
                        message=SimpleNamespace(content="", reasoning_content="internal thought"),
                    ),
                ]
                self.usage = None

        class _ChatCompletions:
            def create(self, **kwargs: object) -> _Response:
                del kwargs
                return _Response()

        client = SimpleNamespace(chat=SimpleNamespace(completions=_ChatCompletions()))
        config = BaselineConfig(
            repo_dir=tmp_path,
            model="test-model",
            api_base_url="http://127.0.0.1:9999",
            api_key="sk-test",
            # BAS-19: the shared default enables the reasoning fallback; disable it here.
            extra={
                "baseline_openai_enable_thinking": False,
                "baseline_openai_allow_reasoning_fallback": False,
            },
        )

        with pytest.raises(OpenAICompatResponseError, match="reasoning but no final answer"):
            adapter._generate_text(
                client=client,
                model="test-model",
                prompt="hello",
                max_tokens=32,
                config=config,
            )

    def test_env_client_returns_compatibility_error_payload_for_reasoning_only_response(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "model": "test-model",
                    "prompt": "hello",
                    "max_tokens": 32,
                    "base_url": "http://127.0.0.1:9999",
                    "api_key": "sk-test",
                    "timeout_sec": 7,
                    "baseline_openai_enable_thinking": False,
                    "baseline_openai_allow_reasoning_fallback": False,
                }
            ),
            encoding="utf-8",
        )

        class _Response:
            def __init__(self) -> None:
                self.choices = [
                    SimpleNamespace(
                        message=SimpleNamespace(content="", reasoning_content="internal thought"),
                    ),
                ]
                self.usage = None

        class _ChatCompletions:
            def create(self, **kwargs: object) -> _Response:
                del kwargs
                return _Response()

        class _OpenAI:
            def __init__(self, **kwargs: object) -> None:
                del kwargs
                self.chat = SimpleNamespace(completions=_ChatCompletions())

        monkeypatch.setattr(
            env_client, "import_module", lambda name: SimpleNamespace(OpenAI=_OpenAI)
        )

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert payload["error"] == "openai_compat_reasoning_only"
        assert "reasoning but no final answer" in payload["compat_message"]
        assert payload["compat_meta"]["empty_content_with_reasoning"] is True

    def test_run_repair_includes_adaptive_controller_guidance(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
            timeout_sec=20,
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_decision_reason": "policy_action_snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": [
                        "Crash is localized to function multiply",
                        "Relevant locals: a, b",
                    ],
                    "selected_local_names": ["a", "b"],
                },
                "ast_context": {
                    "path": "buggy.py",
                    "line_span": [1, 2],
                    "symbol_name": "multiply",
                    "node_type": "FunctionDef",
                    "snippet": "def multiply(a, b):\n    return a + b",
                },
                "snapshot_reduction": {
                    "crash_frame_summary": "buggy.py:2 in multiply() -> return a + b"
                },
                "items": [
                    {
                        "id": "e1",
                        "kind": "frame",
                        "locator": "frame:0",
                        "summary": "AssertionError on multiply output",
                        "path": "buggy.py",
                    }
                ],
            },
            "history_bundle": {
                "previous_note": "tests_still_failing",
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "must preserve function contract",
                    }
                ],
                "retry_history_text": "Attempt 1: tests_still_failing",
                "repo_history": {
                    "recent_commits": [{"summary": "fix helper math"}],
                    "crash_file": {"blame": {"summary": "introduced off-by-one bug"}},
                },
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["patch missed crash file"],
            },
            "solver_hints": {
                "retry_plan": "Focus on multiply return operator.",
                "strategy_summary": "First localize the return path, then keep the edit minimal.",
                "decomposition": [
                    {
                        "title": "localize_fix",
                        "question": "Identify the minimal behavior change required at buggy.py:2 in multiply.",
                    }
                ],
            },
        }
        prompts: list[str] = []

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            prompts.append(prompt)
            if "Modification method:" in prompt:
                return "Change the operator from + to *.", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 7,
                "completion_tokens": 8,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert any("Adaptive controller guidance:" in prompt for prompt in prompts)
        assert any("crash_anchor: buggy.py:2 in multiply" in prompt for prompt in prompts)
        assert any("evidence_summary:" in prompt for prompt in prompts)
        assert any("history_lessons:" in prompt for prompt in prompts)
        assert any("ast_focus: buggy.py" in prompt for prompt in prompts)
        assert any("recent_commits: fix helper math" in prompt for prompt in prompts)
        assert any("strategy_summary:" in prompt for prompt in prompts)
        assert any("decomposition: localize_fix:" in prompt for prompt in prompts)
        assert any("retry_plan:" in prompt for prompt in prompts)

    def test_run_repair_uses_fair_v1_hybrid_guidance_block(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=1,
            timeout_sec=20,
            extra={"baseline_hybrid_guidance_mode": "fair_v1"},
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_decision_reason": "policy_action_snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": [
                        "Edit the multiply return operator",
                        "Keep the function contract unchanged",
                    ],
                    "selected_local_names": ["a", "b"],
                },
                "ast_context": {
                    "path": "buggy.py",
                    "line_span": [1, 2],
                    "symbol_name": "multiply",
                    "node_type": "FunctionDef",
                    "snippet": "def multiply(a, b):\n    return a + b\n",
                },
                "items": [
                    {
                        "id": "e1",
                        "kind": "frame",
                        "locator": "frame:0",
                        "summary": "AssertionError on multiply output",
                        "path": "buggy.py",
                    }
                ],
            },
            "history_bundle": {
                "previous_note": "tests_still_failing",
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "must preserve function contract",
                    }
                ],
                "retry_history_text": "Attempt 1: tests_still_failing",
                "repo_history": {
                    "recent_commits": [{"summary": "fix helper math"}],
                    "crash_file": {"blame": {"summary": "introduced off-by-one bug"}},
                },
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["patch missed crash file", "preserve signature"],
            },
            "solver_hints": {
                "retry_plan": "Focus on multiply return operator.",
                "strategy_summary": "should not appear",
                "decomposition": [
                    {
                        "title": "localize_fix",
                        "question": "should not appear",
                    }
                ],
            },
        }
        prompts: list[str] = []

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            prompts.append(prompt)
            if "Modification method:" in prompt:
                return "Change the operator from + to *.", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 7,
                "completion_tokens": 8,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert any("Adaptive controller guidance:" in prompt for prompt in prompts)
        assert sum("crash_anchor: buggy.py:2 in multiply" in prompt for prompt in prompts) == 2
        assert any("verifier: status=previous_verifier_rejected" in prompt for prompt in prompts)
        assert any("retry_plan: Focus on multiply return operator." in prompt for prompt in prompts)
        assert not any("ast_focus:" in prompt for prompt in prompts)
        assert not any("recent_commits:" in prompt for prompt in prompts)
        assert not any("strategy_summary:" in prompt for prompt in prompts)
        assert not any("decomposition:" in prompt for prompt in prompts)

    def test_run_repair_uses_fixed_v1_shared_evidence_block(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=1,
            timeout_sec=20,
            extra={
                "baseline_hybrid_guidance_mode": "fixed_v1",
                "baseline_intervenor_teacher_guidance_max_chars": 4000,
                "baseline_intervenor_learner_guidance_max_chars": 4000,
            },
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_decision_reason": "policy_action_snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:2 in multiply",
                    "top_facts": [
                        "Edit the multiply return operator",
                        "Keep the function contract unchanged",
                    ],
                    "selected_local_names": ["a", "b"],
                },
                "ast_context": {
                    "path": "buggy.py",
                    "line_span": [1, 2],
                    "symbol_name": "multiply",
                    "node_type": "FunctionDef",
                    "snippet": "def multiply(a, b):\n    return a + b\n",
                },
                "items": [
                    {
                        "id": "e1",
                        "kind": "frame",
                        "locator": "frame:0",
                        "summary": "AssertionError on multiply output",
                        "path": "buggy.py",
                    }
                ],
                "snapshot_reduction": {
                    "crash_frame_summary": "buggy.py:2 in multiply() a=3 b=4",
                    "selected_frame_count": 1,
                    "omitted_frame_count": 0,
                    "selected_frames": [],
                    "selected_locals": [{"name": "a", "summary": "3"}],
                },
                "snapshot_full_text": json.dumps(
                    {
                        "frames": [
                            {
                                "file_rel": "buggy.py",
                                "line": 2,
                                "locals": {"a": "3", "b": "4"},
                            }
                        ]
                    },
                    indent=2,
                ),
            },
            "history_bundle": {
                "previous_note": "tests_still_failing",
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "must preserve function contract",
                    }
                ],
                "retry_history_text": "Attempt 1: tests_still_failing",
                "repo_history": {
                    "recent_commits": [{"summary": "fix helper math"}],
                    "crash_file": {"blame": {"summary": "introduced off-by-one bug"}},
                },
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["patch missed crash file", "preserve signature"],
            },
            "solver_hints": {
                "retry_plan": "Focus on multiply return operator.",
                "strategy_summary": "should not appear",
                "decomposition": [{"title": "localize_fix", "question": "should not appear"}],
            },
        }
        prompts: list[str] = []

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            prompts.append(prompt)
            if "Modification method:" in prompt:
                return "Change the operator from + to *.", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 7,
                "completion_tokens": 8,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        guidance_payloads = []
        for prompt in prompts:
            marker = "Adaptive controller guidance:\n"
            assert marker in prompt
            payload = prompt.split(marker, 1)[1]
            payload = payload.split("\n\nYOUR TASK:", 1)[0]
            guidance_payloads.append(payload)
        assert len(set(guidance_payloads)) == 1
        assert all(
            "crash_anchor: buggy.py:2 in multiply" in payload for payload in guidance_payloads
        )
        assert all("evidence_summary:" in payload for payload in guidance_payloads)
        assert all(
            "ast_focus: buggy.py lines=(1, 2) symbol=multiply" in payload
            for payload in guidance_payloads
        )
        assert all("ast_snippet:" in payload for payload in guidance_payloads)
        assert all("selected_evidence:" in payload for payload in guidance_payloads)
        assert all("snapshot_full:" in payload for payload in guidance_payloads)
        assert all('"file_rel": "buggy.py"' in payload for payload in guidance_payloads)
        assert all('"line": 2' in payload for payload in guidance_payloads)
        assert all("verifier" not in payload for payload in guidance_payloads)
        assert all("history_lessons:" not in payload for payload in guidance_payloads)
        assert all("retry_plan:" not in payload for payload in guidance_payloads)
        assert all("strategy_summary:" not in payload for payload in guidance_payloads)
        assert all("decomposition:" not in payload for payload in guidance_payloads)

    def test_run_repair_requires_python_args(self, simple_case: Path, tmp_path: Path) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(repo_dir=repo)
        adapter.setup(config)

        result = adapter.run_repair(adapter.convert_case(simple_case).to_dict(), config)

        assert result.success is False
        assert result.meta.get("error") == "missing_python_args"

    def test_run_repair_skips_inprocess_client_when_env_python_is_configured(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        fake_python = tmp_path / "fake-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)

        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            max_attempts=2,
            timeout_sec=20,
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()

        prompts: list[str] = []
        responses = iter(
            [
                (
                    "Change the operator from + to *.",
                    {"prompt_tokens": 10, "completion_tokens": 5},
                    0.0,
                ),
                (
                    "def multiply(a, b):\n    return a * b\n",
                    {"prompt_tokens": 7, "completion_tokens": 8},
                    0.0,
                ),
            ]
        )

        def _unexpected_build_client(config: BaselineConfig) -> object:
            raise AssertionError("run_repair should not build an in-process client")

        def _fake_env_generate_text(
            *,
            python_executable: Path,
            model: str,
            prompt: str,
            max_tokens: int,
            config: BaselineConfig,
            timeout_sec: float | None = None,
        ) -> tuple[str, dict[str, int], float]:
            prompts.append(prompt)
            assert python_executable == fake_python
            assert model == "test-model"
            return next(responses)

        monkeypatch.setattr(adapter, "_build_client", _unexpected_build_client)
        monkeypatch.setattr(adapter, "_generate_text_via_env_python", _fake_env_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.repaired_code == "def multiply(a, b):\n    return a * b"
        assert result.iterations_used == 1
        assert result.token_usage == {
            "prompt_tokens": 17,
            "completion_tokens": 13,
            "total_tokens": 30,
        }
        assert any("Teacher prompt" in prompt for prompt in prompts)
        assert any("Learner prompt" in prompt for prompt in prompts)

    def test_run_repair_tracks_empty_learner_responses(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
            timeout_sec=20,
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        responses = iter(
            [
                ("Change the operator from + to *.", {"prompt_tokens": 10, "completion_tokens": 5}),
                ("", {"prompt_tokens": 7, "completion_tokens": 8}),
            ]
        )

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            return next(responses)

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.meta["error"] == "learner_empty_response"
        metrics = result.meta["metrics"]
        self._assert_attempt_metric_shape(metrics)
        assert metrics["learner_empty_response_count"] == 1
        assert metrics["attempts_with_postcheck_count"] == 0
        assert metrics["usage_reported_call_count"] == 2
        assert metrics["usage_missing_call_count"] == 0
        assert metrics["reported_tokens_total"] == 30

    def test_run_repair_retries_on_syntax_broken_learner_output(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
            timeout_sec=20,
        )
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        prompts: list[str] = []
        responses = iter(
            [
                ("Change the operator from + to *.", {"prompt_tokens": 10, "completion_tokens": 5}),
                (
                    "def multiply(a, b)\n    return a * b\n",
                    {"prompt_tokens": 7, "completion_tokens": 8},
                ),
                (
                    "Add the missing colon and return only Python.",
                    {"prompt_tokens": 9, "completion_tokens": 4},
                ),
                (
                    "def multiply(a, b):\n    return a * b\n",
                    {"prompt_tokens": 8, "completion_tokens": 7},
                ),
            ]
        )

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            del client, model, max_tokens, config
            prompts.append(prompt)
            return next(responses)

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.iterations_used == 2
        assert result.repaired_code == "def multiply(a, b):\n    return a * b"
        metrics = result.meta["metrics"]
        self._assert_attempt_metric_shape(metrics)
        assert metrics["attempts_with_postcheck_count"] == 2
        assert metrics["learner_empty_response_count"] == 0
        assert len(prompts) == 4
        assert any(
            marker in prompts[2] for marker in ("SyntaxError", "expected ':'", "invalid syntax")
        )

    def test_run_repair_case_already_passed_uses_shared_metric_shape(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case_dir = tmp_path / "passing_case"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text(
            "def multiply(a, b):\n    return a * b\n",
            encoding="utf-8",
        )
        (case_dir / "test_buggy.py").write_text(
            "from buggy import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
            encoding="utf-8",
        )
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(repo_dir=repo, model="test-model", max_attempts=2, timeout_sec=20)
        adapter.setup(config)
        case_input = adapter.convert_case(
            case_dir,
            case_context=BaselineCaseContext(
                case_id="passing_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        monkeypatch.setattr(adapter, "_build_client", lambda config: object())

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.iterations_used == 0
        metrics = result.meta["metrics"]
        self._assert_attempt_metric_shape(metrics)
        assert metrics["attempts_with_postcheck_count"] == 0
        assert metrics["learner_empty_response_count"] == 0

    def test_run_repair_max_attempts_exhausted_uses_shared_metric_shape(
        self,
        simple_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_intervenor_repo(tmp_path)
        adapter = IntervenorAdapter()
        config = BaselineConfig(repo_dir=repo, model="test-model", max_attempts=1, timeout_sec=20)
        adapter.setup(config)
        case_input = adapter.convert_case(
            simple_case,
            case_context=BaselineCaseContext(
                case_id="my_case",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
            ),
        ).to_dict()
        responses = iter(
            [
                ("Change the operator from + to -.", {"prompt_tokens": 10, "completion_tokens": 5}),
                (
                    "def multiply(a, b):\n    return a - b\n",
                    {"prompt_tokens": 7, "completion_tokens": 8},
                ),
            ]
        )

        def _fake_generate_text(
            *, client, model: str, prompt: str, max_tokens: int, config=None, **_deadline_kwargs
        ):
            return next(responses)

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.meta["error"] == "max_attempts_exhausted"
        metrics = result.meta["metrics"]
        self._assert_attempt_metric_shape(metrics)
        assert metrics["attempts_with_postcheck_count"] == 1
        assert metrics["learner_empty_response_count"] == 0

    def test_extract_diff_with_repair(self, simple_case: Path) -> None:
        result = BaselineResult(
            repaired_code="def multiply(a, b):\n    return a * b\n",
            success=True,
            iterations_used=1,
        )
        diff = IntervenorAdapter().extract_diff(simple_case, result)
        assert diff is not None
        assert "+    return a * b" in diff


def test_bas10_generation_failure_keeps_candidate_and_usage() -> None:
    from baselines.adapter_base import BaselineResult
    from baselines.intervenor.adapter import _keep_progress_on_generation_failure

    failure = BaselineResult(
        repaired_code=None,
        success=False,
        iterations_used=1,
        token_usage={"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        meta={"error": "openai_compat_reasoning_only", "compat_message": "x", "metrics": {}},
    )
    kept = _keep_progress_on_generation_failure(
        exc=TimeoutError("read timed out"),
        current_code="def f():\n    return 2\n",
        original_code="def f():\n    return 1\n",
        failure=failure,
    )
    assert kept.repaired_code == "def f():\n    return 2\n"
    assert kept.token_usage["total_tokens"] == 35
    assert kept.meta["error"] == "generation_failed"
    assert "TimeoutError" in kept.meta["generation_exc"]
    assert "compat_message" not in kept.meta


def test_bas17_usage_extra_body_only_for_endpoints_that_support_it() -> None:
    from baselines.openai_compat import usage_include_extra_body

    assert usage_include_extra_body("https://openrouter.ai/api/v1") == {"usage": {"include": True}}
    assert usage_include_extra_body("http://127.0.0.1:8080/v1") == {}
    assert usage_include_extra_body("https://api.openai.com/v1") == {}
    assert usage_include_extra_body("https://evil-openrouter.ai.example.com") == {}
