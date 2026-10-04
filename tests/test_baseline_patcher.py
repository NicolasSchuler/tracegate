"""Tests for the baseline patcher infrastructure.

Exercises BaselinePatcher, the adapter registry, and the factory function
without requiring external repos or LLM access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import evals.patchers.baseline_patcher as baseline_patcher_module
from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineCaseIncompatibleError,
    BaselineConfig,
    BaselineFileEdit,
    BaselineResult,
)
from baselines.case_schemas import IntervenorCaseInput
from baselines.contracts import BASELINE_CONTRACT_VERSION, BaselineContract
from evals.eval_patches import apply_diff
from evals.patchers.base import ContextRequestResult
from evals.patchers.baseline_patcher import (
    BaselinePatcher,
    _baseline_hybrid_guidance_metadata,
    _build_adapter_registry,
    _exact_baseline_char_totals,
    _infer_repair_contract,
    available_baselines,
    build_baseline_patcher,
    get_adapter_registry,
    resolve_effective_baseline_model,
)

# ---------------------------------------------------------------------------
# Stub adapter for testing
# ---------------------------------------------------------------------------


class _StubAdapter(BaselineAdapter):
    """Minimal adapter that returns deterministic results."""

    _repair_result: BaselineResult
    seen_case_context: BaselineCaseContext | None
    seen_case_input: dict[str, Any] | None
    saw_progress_callback: bool

    def __init__(
        self,
        repair_result: BaselineResult | None = None,
        *,
        name: str = "stub",
    ) -> None:
        self._name = name
        self._repair_result = repair_result or BaselineResult(
            repaired_code="def stub():\n    return 42\n",
            success=True,
            iterations_used=1,
        )
        self.seen_case_context = None
        self.seen_case_input = None
        self.saw_progress_callback = False

    @property
    def name(self) -> str:
        return self._name

    def setup(self, config: BaselineConfig) -> None:
        pass

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> dict[str, Any]:
        self.seen_case_context = case_context
        buggy = (case_dir / "buggy.py").read_text(encoding="utf-8")
        return {"task_id": case_dir.name, "buggy_code": buggy}

    def run_repair(self, case_input: dict[str, Any], config: BaselineConfig) -> BaselineResult:
        self.seen_case_input = dict(case_input)
        self.saw_progress_callback = self._current_progress_callback() is not None
        return self._repair_result


@pytest.fixture()
def stub_case(tmp_path: Path) -> Path:
    """Minimal case directory for stub adapter tests."""
    case_dir = tmp_path / "stub_case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def stub():\n    return 0\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import stub\n\ndef test_stub():\n    assert stub() == 42\n",
        encoding="utf-8",
    )
    (case_dir / "case.json").write_text(
        json.dumps({"description": "stub", "category": "logic", "difficulty": "easy"}),
        encoding="utf-8",
    )
    return case_dir


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for the lazy adapter registry."""

    def test_registry_returns_dict(self) -> None:
        reg = get_adapter_registry()
        assert isinstance(reg, dict)

    def test_ldb_in_registry(self) -> None:
        reg = get_adapter_registry()
        assert "ldb" in reg

    def test_intervenor_in_registry(self) -> None:
        reg = get_adapter_registry()
        assert "intervenor" in reg

    def test_available_baselines_returns_sorted(self) -> None:
        names = available_baselines()
        assert names == sorted(names)
        assert "ldb" in names
        assert "intervenor" in names
        assert "tracecoder" in names
        assert "lambda_rlm" in names
        assert "rlm" in names

    def test_build_adapter_registry_tolerates_missing_optional_adapters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {
                "baselines.ldb.adapter",
                "baselines.intervenor.adapter",
                "baselines.tracecoder.adapter",
                "baselines.lambda_rlm.adapter",
                "baselines.rlm.adapter",
            }:
                raise ImportError("missing optional adapter")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        assert _build_adapter_registry() == {}


# ---------------------------------------------------------------------------
# BaselinePatcher tests
# ---------------------------------------------------------------------------


class TestBaselinePatcher:
    """Tests for the BaselinePatcher wrapper."""

    def test_name_includes_adapter_name(self) -> None:
        adapter = _StubAdapter()
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)
        assert patcher.name == "stub"

    def test_probe_server_ready_uses_default_api_key_and_effective_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter(name="tracecoder")
        adapter.default_model = "adapter-default-model"
        config = BaselineConfig(
            model="",
            api_base_url="http://127.0.0.1:4321/v1",
            api_key="",
            api_timeout_sec=12.5,
        )
        patcher = BaselinePatcher(adapter=adapter, config=config)
        observed: dict[str, Any] = {}

        monkeypatch.setattr(baseline_patcher_module, "_default_api_key", lambda: "env-key")

        def _fake_probe_server_ready(*, base_url, api_key, timeout_sec, http2):
            observed.update(
                {
                    "base_url": base_url,
                    "api_key": api_key,
                    "timeout_sec": timeout_sec,
                    "http2": http2,
                }
            )
            return True, {}

        monkeypatch.setattr(
            baseline_patcher_module,
            "_probe_server_ready",
            _fake_probe_server_ready,
        )

        ready, meta = patcher.probe_server_ready()

        assert ready is True
        assert observed == {
            "base_url": "http://127.0.0.1:4321/v1",
            "api_key": "env-key",
            "timeout_sec": 12.5,
            "http2": False,
        }
        assert meta["base_url"] == "http://127.0.0.1:4321/v1"
        assert meta["model"] == "adapter-default-model"

    def test_probe_server_ready_preserves_probe_metadata_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter()
        config = BaselineConfig(
            model="configured-model",
            api_base_url="http://127.0.0.1:9999/v1",
            api_key="configured-key",
        )
        patcher = BaselinePatcher(adapter=adapter, config=config)

        monkeypatch.setattr(
            baseline_patcher_module,
            "_probe_server_ready",
            lambda **_: (
                False,
                {
                    "base_url": "probe-base-url",
                    "model": "probe-model",
                    "status": "warming",
                },
            ),
        )

        ready, meta = patcher.probe_server_ready()

        assert ready is False
        assert meta["base_url"] == "probe-base-url"
        assert meta["model"] == "probe-model"
        assert meta["status"] == "warming"

    def test_request_context_uses_cached_openai_helper_delegate(
        self,
        stub_case: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter(name="tracecoder")
        adapter.default_model = "adapter-default-model"
        config = BaselineConfig(
            model="",
            api_base_url="http://127.0.0.1:4321/v1",
            api_key="configured-key",
            api_timeout_sec=12.5,
            extra={
                "baseline_openai_compat_profile": "lmstudio_model_yaml",
                "baseline_openai_allow_reasoning_fallback": False,
                "baseline_openai_enable_thinking": False,
            },
        )
        patcher = BaselinePatcher(adapter=adapter, config=config)
        observed_init: list[dict[str, Any]] = []
        observed_calls: list[dict[str, Any]] = []

        class _FakeOpenAICompatPatcher:
            def __init__(self, **kwargs: Any) -> None:
                observed_init.append(dict(kwargs))

            def request_context(
                self,
                *,
                prompt: str,
                context_dir: Path,
                cwd: Path,
                strict_json_first: bool = False,
            ) -> ContextRequestResult:
                observed_calls.append(
                    {
                        "prompt": prompt,
                        "context_dir": context_dir,
                        "cwd": cwd,
                        "strict_json_first": strict_json_first,
                    }
                )
                return ContextRequestResult(
                    request_json={"need_more_context": True, "requests": []},
                    raw_response='{"need_more_context":true}',
                    meta={"source": "fake_helper"},
                )

        monkeypatch.setattr(
            baseline_patcher_module,
            "OpenAICompatPatcher",
            _FakeOpenAICompatPatcher,
        )

        context_dir = stub_case / "context"
        context_dir.mkdir()
        result_one = patcher.request_context(
            prompt="first prompt",
            context_dir=context_dir,
            cwd=stub_case,
            strict_json_first=True,
        )
        result_two = patcher.request_context(
            prompt="second prompt",
            context_dir=context_dir,
            cwd=stub_case,
            strict_json_first=False,
        )

        assert patcher.supports_request_context() is True
        assert patcher.supports_context_compaction() is False
        assert patcher.supports_strict_json_first() is True
        assert len(observed_init) == 1
        assert observed_init[0] == {
            "base_url": "http://127.0.0.1:4321/v1",
            "model": "adapter-default-model",
            "api_key": "configured-key",
            "timeout_sec": 12.5,
            "compat_profile": "lmstudio_model_yaml",
            "allow_reasoning_fallback": False,
            "enable_thinking": False,
            "name": "tracecoder_stage_a_helper",
        }
        assert observed_calls == [
            {
                "prompt": "first prompt",
                "context_dir": context_dir,
                "cwd": stub_case,
                "strict_json_first": True,
            },
            {
                "prompt": "second prompt",
                "context_dir": context_dir,
                "cwd": stub_case,
                "strict_json_first": False,
            },
        ]
        assert result_one is not None
        assert result_one.meta == {"source": "fake_helper"}
        assert result_two is not None

    def test_propose_patch_returns_diff(self, stub_case: Path) -> None:
        adapter = _StubAdapter()
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)
        assert result.diff is not None
        assert "return 42" in result.diff
        assert result.meta is not None
        assert result.meta.get("baseline") == "stub"
        assert result.meta.get("baseline_success") is True
        assert result.meta.get("repair_scope") == "single_file"
        assert result.meta.get("repair_targets") == ["buggy.py"]
        assert result.meta.get("contract_version") == BASELINE_CONTRACT_VERSION
        assert result.meta.get("config_validation_mode") == "strict"

    def test_propose_patch_no_diff_when_same(self, stub_case: Path) -> None:
        same_code = (stub_case / "buggy.py").read_text(encoding="utf-8")
        adapter = _StubAdapter(repair_result=BaselineResult(repaired_code=same_code, success=False))
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)
        assert result.diff is None

    def test_propose_patch_none_repair(self, stub_case: Path) -> None:
        adapter = _StubAdapter(repair_result=BaselineResult(repaired_code=None, success=False))
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)
        assert result.diff is None

    def test_propose_patch_conversion_failure(self, tmp_path: Path) -> None:
        """Patcher returns structured error when case_dir has no buggy.py."""
        empty_case = tmp_path / "empty"
        empty_case.mkdir()

        adapter = _StubAdapter()
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)

        result = patcher.propose_patch(prompt="ignored", context_dir=empty_case, cwd=empty_case)
        assert result.diff is None
        assert result.meta is not None
        assert result.meta.get("error") == "case_conversion_failed"

    def test_propose_patch_case_incompatible(
        self,
        stub_case: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        def _raise_case_incompatible(*args, **kwargs):
            raise BaselineCaseIncompatibleError(
                reason="subprocess_task_unsupported",
                detail="case launches a subprocess",
            )

        monkeypatch.setattr(adapter, "convert_case", _raise_case_incompatible)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "case_incompatible"
        assert result.meta["baseline_supported"] is False
        assert result.meta["compatibility_reason"] == "subprocess_task_unsupported"
        assert result.meta["compatibility_detail"] == "case launches a subprocess"

    def test_propose_patch_case_incompatible_without_detail(
        self,
        stub_case: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        def _raise_case_incompatible(*args, **kwargs):
            raise BaselineCaseIncompatibleError(reason="unsupported_case")

        monkeypatch.setattr(adapter, "convert_case", _raise_case_incompatible)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "case_incompatible"
        assert result.meta["compatibility_reason"] == "unsupported_case"
        assert "compatibility_detail" not in result.meta

    def test_propose_patch_accepts_typed_case_input(self, stub_case: Path) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        def _typed_case(
            case_dir: Path,
            *,
            case_context: BaselineCaseContext | None = None,
        ) -> IntervenorCaseInput:
            del case_dir, case_context
            return IntervenorCaseInput(
                task_id="typed",
                entry_point="buggy.py",
                prompt="fix the bug",
                buggy_code="def stub():\n    return 0\n",
            )

        adapter.convert_case = _typed_case

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is not None
        assert adapter.seen_case_input is not None
        assert adapter.seen_case_input["task_id"] == "typed"
        assert adapter.seen_case_input["entry_point"] == "buggy.py"
        assert adapter.seen_case_input["prompt"] == "fix the bug"

    def test_propose_patch_rejects_invalid_case_input_type(self, stub_case: Path) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())
        adapter.convert_case = lambda *args, **kwargs: 123

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "case_conversion_failed"
        assert "BaseCaseInput or dict" in result.meta["exc"]

    def test_propose_patch_repair_failure(
        self,
        stub_case: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        def _raise_repair_failed(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(adapter, "run_repair", _raise_repair_failed)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "repair_failed"
        assert result.meta["exc"] == "RuntimeError: boom"

    def test_propose_patch_forwards_and_clears_progress_callback(self, stub_case: Path) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        patcher.set_progress_callback(lambda _event, _payload: None)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is not None
        assert adapter.saw_progress_callback is True
        assert adapter._current_progress_callback() is None

    def test_propose_patch_records_iterations(self, stub_case: Path) -> None:
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 42\n",
                success=True,
                iterations_used=3,
                token_usage={"prompt_tokens": 100, "completion_tokens": 50},
                meta={
                    "metrics": {
                        "llm_call_count": 2,
                        "llm_duration_sec": 1.25,
                        "eval_duration_sec": 0.75,
                        "internal_total_duration_sec": 2.5,
                        "orchestration_duration_sec": 0.5,
                        "usage_available": True,
                        "usage_reported_call_count": 2,
                        "usage_missing_call_count": 0,
                        "llm_timeout_count": 1,
                        "reported_tokens_prompt": 100,
                        "reported_tokens_completion": 50,
                        "reported_tokens_total": 150,
                        "prompt_chars_total": 240,
                        "completion_chars_total": 60,
                        "chars_total": 300,
                        "reported_chars_prompt": 240,
                        "reported_chars_completion": 60,
                        "reported_chars_total": 300,
                        "stopped_reason": "Debug cycle completed",
                        "method_instrumentation": True,
                        "method_two_step_repair": False,
                        "direct_gen_passed": False,
                        "debug_added_value": True,
                        "teacher_duration_sec_total": 0.8,
                        "learner_duration_sec_total": 0.4,
                        "postcheck_duration_sec_total": 0.2,
                        "learner_empty_response_count": 1,
                        "attempts_with_postcheck_count": 2,
                        "is_passing": True,
                        "test_feedback_steps": 3,
                        "test_feedback_nonempty_steps": 2,
                        "test_feedback_improvement_steps": 1,
                        "test_feedback_regression_steps": 0,
                        "test_feedback_stagnation_steps": 1,
                        "test_feedback_final_fail_count": 0,
                        "usage_source": "provider_reported",
                        "last_llm_failure_kind": "api_timeout",
                        "last_llm_failure_stage": "analysis_planning",
                    }
                },
            )
        )
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)
        assert result.meta is not None
        assert result.meta.get("iterations_used") == 3
        assert result.meta.get("usage") == {"prompt_tokens": 100, "completion_tokens": 50}
        assert result.meta.get("llm_call_count") == 2
        assert result.meta.get("prompt_chars_total") == 240
        assert result.meta.get("completion_chars_total") == 60
        assert result.meta.get("chars_input") == 240
        assert result.meta.get("chars_output") == 60
        assert result.meta.get("chars_total") == 300
        assert result.meta.get("baseline_metrics") == {
            "llm_call_count": 2,
            "llm_duration_sec": 1.25,
            "eval_duration_sec": 0.75,
            "internal_total_duration_sec": 2.5,
            "orchestration_duration_sec": 0.5,
            "usage_available": True,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 0,
            "llm_timeout_count": 1,
            "reported_tokens_prompt": 100,
            "reported_tokens_completion": 50,
            "reported_tokens_total": 150,
            "prompt_chars_total": 240,
            "completion_chars_total": 60,
            "chars_total": 300,
            "reported_chars_prompt": 240,
            "reported_chars_completion": 60,
            "reported_chars_total": 300,
            "stopped_reason": "Debug cycle completed",
            "method_instrumentation": True,
            "method_two_step_repair": False,
            "direct_gen_passed": False,
            "debug_added_value": True,
            "teacher_duration_sec_total": 0.8,
            "learner_duration_sec_total": 0.4,
            "postcheck_duration_sec_total": 0.2,
            "learner_empty_response_count": 1,
            "attempts_with_postcheck_count": 2,
            "is_passing": True,
            "test_feedback_steps": 3,
            "test_feedback_nonempty_steps": 2,
            "test_feedback_improvement_steps": 1,
            "test_feedback_regression_steps": 0,
            "test_feedback_stagnation_steps": 1,
            "test_feedback_final_fail_count": 0,
            "usage_source": "provider_reported",
            "last_llm_failure_kind": "api_timeout",
            "last_llm_failure_stage": "analysis_planning",
        }

    def test_propose_patch_preserves_partial_usage_baseline_metrics(self, stub_case: Path) -> None:
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 42\n",
                success=False,
                iterations_used=2,
                token_usage={},
                meta={
                    "metrics": {
                        "llm_call_count": 3,
                        "usage_available": False,
                        "usage_reported_call_count": 2,
                        "usage_missing_call_count": 1,
                        "reported_tokens_prompt": 20,
                        "reported_tokens_completion": 6,
                        "reported_tokens_total": 26,
                        "prompt_chars_total": 180,
                        "completion_chars_total": 54,
                        "chars_total": 234,
                        "reported_chars_prompt": 180,
                        "reported_chars_completion": 54,
                        "reported_chars_total": 234,
                        "usage_source": "partial_provider_reported",
                    }
                },
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.meta is not None
        assert "usage" not in result.meta
        assert result.meta.get("prompt_chars_total") == 180
        assert result.meta.get("completion_chars_total") == 54
        assert result.meta.get("chars_total") == 234
        assert result.meta.get("baseline_metrics") == {
            "llm_call_count": 3,
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 20,
            "reported_tokens_completion": 6,
            "reported_tokens_total": 26,
            "prompt_chars_total": 180,
            "completion_chars_total": 54,
            "chars_total": 234,
            "reported_chars_prompt": 180,
            "reported_chars_completion": 54,
            "reported_chars_total": 234,
            "usage_source": "partial_provider_reported",
        }

    def test_propose_patch_uses_effective_default_model(self, stub_case: Path) -> None:
        adapter = _StubAdapter(name="intervenor")
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig(model=""))

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.meta is not None
        assert result.meta.get("model") == "gpt-4o-mini"

    def test_resolve_effective_baseline_model_prefers_adapter_default(self) -> None:
        adapter = _StubAdapter()
        adapter.default_model = "adapter-model"

        assert (
            resolve_effective_baseline_model(
                baseline_name="stub",
                configured_model="",
                adapter=adapter,
            )
            == "adapter-model"
        )

    def test_resolve_effective_baseline_model_falls_back_to_legacy_table_when_adapter_blank(
        self,
    ) -> None:
        adapter = _StubAdapter(name="intervenor")

        assert (
            resolve_effective_baseline_model(
                baseline_name="intervenor",
                configured_model="",
                adapter=adapter,
            )
            == "gpt-4o-mini"
        )

    def test_resolve_effective_baseline_model_falls_back_to_legacy_table_without_adapter(
        self,
    ) -> None:
        assert (
            resolve_effective_baseline_model(
                baseline_name="intervenor",
                configured_model="",
                adapter=None,
            )
            == "gpt-4o-mini"
        )

    def test_propose_patch_prefers_patch_diff(self, stub_case: Path) -> None:
        custom_diff = (
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def stub():\n"
            "-    return 0\n"
            "+    return 42\n"
        )
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 41\n",
                patch_diff=custom_diff,
                success=True,
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff == custom_diff
        assert result.meta is not None
        assert result.meta.get("repair_scope") == "patch_diff"

    def test_propose_patch_preserves_explicit_repair_targets_from_result_meta(
        self,
        stub_case: Path,
    ) -> None:
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 42\n",
                success=True,
                meta={
                    "repair_scope": "custom_contract",
                    "repair_targets": [" helper.py ", "", "buggy.py"],
                },
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["repair_scope"] == "custom_contract"
        assert result.meta["repair_targets"] == ["buggy.py", "helper.py"]

    def test_propose_patch_records_baseline_metrics_without_exact_char_totals(
        self,
        stub_case: Path,
    ) -> None:
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 42\n",
                success=True,
                meta={
                    "metrics": {
                        "usage_available": True,
                        "llm_call_count": "three",
                    }
                },
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_metrics"] == {"usage_available": True}
        assert "llm_call_count" not in result.meta
        assert "prompt_chars_total" not in result.meta

    def test_propose_patch_synthesizes_multifile_diff(self, stub_case: Path) -> None:
        (stub_case / "helper.py").write_text("VALUE = 0\n", encoding="utf-8")
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code=None,
                file_edits=[
                    BaselineFileEdit(path="buggy.py", content="def stub():\n    return 42\n"),
                    BaselineFileEdit(path="helper.py", content=None),
                    BaselineFileEdit(path="extra.py", content="ENABLED = True\n"),
                ],
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is not None
        assert "diff --git a/buggy.py b/buggy.py" in result.diff
        assert "diff --git a/helper.py b/helper.py" in result.diff
        assert "diff --git a/extra.py b/extra.py" in result.diff
        assert result.meta is not None
        assert result.meta.get("repair_scope") == "multi_file"
        assert result.meta.get("repair_targets") == ["buggy.py", "extra.py", "helper.py"]

    def test_propose_patch_reports_multifile_scope_even_when_patch_diff_is_authoritative(
        self,
        stub_case: Path,
    ) -> None:
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 0\n",
                file_edits=[BaselineFileEdit(path="helper.py", content="VALUE = 1\n")],
                patch_diff=(
                    "diff --git a/buggy.py b/buggy.py\n"
                    "--- a/buggy.py\n"
                    "+++ b/buggy.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def stub():\n"
                    "-    return 0\n"
                    "+    return 42\n"
                ),
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is not None
        assert "+    return 42" in result.diff
        assert result.meta is not None
        assert result.meta["repair_scope"] == "single_file"
        assert result.meta["repair_targets"] == ["helper.py"]

    def test_propose_patch_rejects_invalid_multifile_paths(self, stub_case: Path) -> None:
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code=None,
                file_edits=[BaselineFileEdit(path="../escape.py", content="oops\n")],
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta.get("error") == "diff_extraction_failed"

    def test_exact_baseline_char_totals_fall_back_to_reported_chars(self) -> None:
        assert _exact_baseline_char_totals(
            {
                "reported_chars_prompt": 180,
                "reported_chars_completion": 54,
            }
        ) == (180, 54)

    def test_propose_patch_passes_case_context(self, stub_case: Path, tmp_path: Path) -> None:
        adapter = _StubAdapter()
        config = BaselineConfig()
        patcher = BaselinePatcher(adapter=adapter, config=config)

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text(
            json.dumps(
                {
                    "schema_version": BASELINE_CONTRACT_VERSION,
                    "case_id": "stub_case",
                    "python_args": ["-m", "pytest", "-q", "test_buggy.py"],
                    "timeout_sec": 17,
                    "description": "stub description",
                    "category": "logic",
                    "difficulty": "easy",
                    "snapshot_dependency": "none",
                }
            ),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert adapter.seen_case_context is not None
        assert adapter.seen_case_context.case_id == "stub_case"
        assert adapter.seen_case_context.python_args == ("-m", "pytest", "-q", "test_buggy.py")
        assert adapter.seen_case_context.timeout_sec == 17
        assert adapter.seen_case_context.schema_version == BASELINE_CONTRACT_VERSION
        # ATT-02 / D1: dataset labels in an older context file are dropped, not forwarded.
        assert not hasattr(adapter.seen_case_context, "description")
        assert result.meta is not None
        assert result.meta["case_info_policy"] == "neutral_v1"
        assert result.meta["baseline_case_context_issues"] == [
            "hidden_case_fields_dropped:description,category,difficulty,snapshot_dependency"
        ]

    def test_propose_patch_preserves_case_context_schema_version(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text(
            json.dumps(
                {
                    "schema_version": "v0",
                    "case_id": "stub_case",
                    "python_args": ["-m", "pytest", "-q", "test_buggy.py"],
                    "timeout_sec": 17,
                }
            ),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_case_context_schema_version"] == "v0"
        assert adapter.seen_case_context is not None
        assert adapter.seen_case_context.schema_version == "v0"

    def test_propose_patch_reports_malformed_case_context_json(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text('{"case_id"', encoding="utf-8")

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_case_context_status"] == "invalid_json"
        assert result.meta["baseline_case_context_error_code"] == "invalid_json"
        assert result.meta["baseline_case_context_error_type"] == "JSONDecodeError"
        assert result.meta["baseline_case_context_error"].startswith("JSONDecodeError:")
        assert adapter.seen_case_context is None

    def test_propose_patch_reports_non_mapping_case_context_payload(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text("[]", encoding="utf-8")

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_case_context_status"] == "invalid_schema"
        assert result.meta["baseline_case_context_error_code"] == "invalid_payload_type"
        assert result.meta["baseline_case_context_error_type"] == "ValueError"
        assert result.meta["baseline_case_context_error_detail"] == "expected object, got list"
        assert adapter.seen_case_context is None

    def test_propose_patch_loads_case_context_with_schema_warnings(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text(
            json.dumps(
                {
                    "case_id": 17,
                    "python_args": ["-m", "pytest", 3],
                    "timeout_sec": "bad-timeout",
                    "description": ["not", "a", "string"],
                    "category": "logic",
                }
            ),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert adapter.seen_case_context is not None
        assert adapter.seen_case_context.case_id == stub_case.name
        assert adapter.seen_case_context.python_args == ("-m", "pytest")
        assert adapter.seen_case_context.timeout_sec == 0
        assert result.meta is not None
        assert result.meta["baseline_case_context_status"] == "loaded_with_warnings"
        assert result.meta["baseline_case_context_issues"] == [
            "schema_version_missing",
            "case_id_invalid_type:int",
            "python_args_dropped_items:1",
            "timeout_sec_invalid_value:'bad-timeout'",
            "hidden_case_fields_dropped:description,category",
        ]

    def test_propose_patch_loads_case_context_with_blank_schema_and_defaults(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text(
            json.dumps({"schema_version": "   ", "case_id": "stub_case"}),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_case_context_status"] == "loaded_with_warnings"
        assert result.meta["baseline_case_context_issues"] == ["schema_version_blank"]
        assert "baseline_case_context_schema_version" not in result.meta
        assert adapter.seen_case_context is not None
        assert adapter.seen_case_context.python_args == ()
        assert adapter.seen_case_context.timeout_sec == 0
        assert adapter.seen_case_context.schema_version == BASELINE_CONTRACT_VERSION

    def test_propose_patch_loads_case_context_with_invalid_field_types(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "baseline_case_context.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "case_id": "stub_case",
                    "python_args": "-m pytest",
                    "timeout_sec": True,
                }
            ),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_case_context_status"] == "loaded_with_warnings"
        assert result.meta["baseline_case_context_issues"] == [
            "schema_version_invalid_type:int",
            "python_args_invalid_type:str",
            "timeout_sec_invalid_type:bool",
        ]
        assert adapter.seen_case_context is not None
        assert adapter.seen_case_context.python_args == ()
        assert adapter.seen_case_context.timeout_sec == 0
        assert adapter.seen_case_context.schema_version == BASELINE_CONTRACT_VERSION

    def test_build_baseline_patcher_rejects_unknown_extra_in_strict_mode(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "intervenor_repo"
        prompt_dir = repo / "prompt"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "code_teacher.txt").write_text("teacher", encoding="utf-8")
        (prompt_dir / "code_learner.txt").write_text("learner", encoding="utf-8")

        with pytest.raises(ValueError, match=r"unknown config\.extra keys"):
            build_baseline_patcher(
                baseline_name="intervenor",
                repo_dir=repo,
                extra={"unexpected_flag": True},
                strict_extra_config=True,
            )

    def test_build_baseline_patcher_warn_mode_accepts_unknown_extra(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        repo = tmp_path / "intervenor_repo"
        prompt_dir = repo / "prompt"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "code_teacher.txt").write_text("teacher", encoding="utf-8")
        (prompt_dir / "code_learner.txt").write_text("learner", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            patcher = build_baseline_patcher(
                baseline_name="intervenor",
                repo_dir=repo,
                extra={"unexpected_flag": True},
                strict_extra_config=False,
            )
        assert patcher.config.extra["baseline_strict_extra_config"] is False
        assert "ignoring unrecognised" in caplog.text

    def test_propose_patch_loads_adaptive_controller_sidecars(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "llmdebug_controller_packet.json").write_text(
            json.dumps(
                {
                    "schema_version": "v1",
                    "_sidecar_schema_version": "sidecar-v2",
                    "controller_state": {
                        "schema_version": "v1",
                        "helper_action_selected": "snapshot_lite",
                        "helper_action_executed": "snapshot_lite",
                        "helper_decision_reason": "",
                        "helper_outcome": "materialized",
                        "helper_materialized_context": True,
                        "helper_calls_used": 1,
                        "helper_budget_max_calls": 2,
                        "no_progress_streak": 0,
                        "retry_prompt_mode": "delta",
                        "retry_context_mode": "history",
                        "context_compaction_requested": False,
                        "context_compaction_invoked": False,
                        "context_compaction_success": False,
                        "context_compaction_trigger": "not_needed",
                        "prompt_budget": {},
                    },
                    "evidence_bundle": {
                        "schema_version": "v1",
                        "case_id": "stub",
                        "condition": "layered_adaptive_llm",
                        "attempt": 1,
                        "failure_signature": "sig",
                        "selected_ids": [],
                        "new_ids": [],
                        "reused_ids": [],
                        "ast_context": None,
                        "snapshot_reduction": None,
                        "summary": None,
                        "items": [],
                    },
                    "history_bundle": {
                        "schema_version": "v1",
                        "retry_context_mode": "history",
                        "attempt_turns": [],
                        "retry_history_text": "",
                        "retry_history_turns_included": 0,
                        "retry_history_truncated": False,
                        "previous_note": "",
                        "previous_failure_signature": "",
                        "previous_patch_verifier_rejected": False,
                        "previous_patch_verifier_reasons": [],
                        "lessons": [],
                        "repo_history": None,
                        "persistent_session_id": "",
                        "persistent_records_loaded": 0,
                    },
                    "verifier_report": {
                        "schema_version": "v1",
                        "status": "none",
                        "scope": "",
                        "reasons": [],
                        "previous_timed_out": False,
                        "previous_timeout_sec": None,
                        "syntax_cycle": 0,
                        "syntax_max_cycles": 0,
                        "last_error_type": "",
                        "last_error_msg": "",
                        "retry_feedback": "",
                    },
                    "solver_hints": {
                        "tool_routes": [],
                        "retry_plan": "",
                        "recommended_focus": [],
                        "decomposition": [],
                        "strategy_summary": "",
                    },
                }
            ),
            encoding="utf-8",
        )
        (context_dir / "llmdebug_evidence_bundle.json").write_text(
            json.dumps({"schema_version": "v1", "items": [{"id": "e1"}]}),
            encoding="utf-8",
        )
        (context_dir / "llmdebug_history_bundle.json").write_text(
            json.dumps({"schema_version": "v1", "attempt_turns": ["Attempt 1"]}),
            encoding="utf-8",
        )
        (context_dir / "llmdebug_verifier_report.json").write_text(
            json.dumps({"schema_version": "v1", "status": "previous_verifier_rejected"}),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["adaptive_controller_packet_loaded"] is True
        assert result.meta["adaptive_evidence_bundle_loaded"] is True
        assert result.meta["adaptive_history_bundle_loaded"] is True
        assert result.meta["adaptive_verifier_report_loaded"] is True
        assert result.meta["adaptive_sidecar_schema_version"] == "sidecar-v2"
        assert result.meta["baseline_hybrid_guidance_mode"] == "tuned_v1"
        assert isinstance(result.meta["baseline_hybrid_guidance_chars"], int)
        assert result.meta["baseline_hybrid_guidance_sections"] == ["controller_state"]
        assert result.meta["baseline_case_conversion_sec"] >= 0.0
        assert result.meta["baseline_sidecar_load_sec"] >= 0.0
        assert adapter.seen_case_input is not None
        assert adapter.seen_case_input["llmdebug_controller_packet"]["schema_version"] == "v1"
        assert adapter.seen_case_input["llmdebug_evidence"]["items"][0]["id"] == "e1"
        assert adapter.seen_case_input["llmdebug_history"]["attempt_turns"] == ["Attempt 1"]
        assert (
            adapter.seen_case_input["llmdebug_verifier_report"]["status"]
            == "previous_verifier_rejected"
        )

    def test_propose_patch_reports_corrupted_adaptive_sidecars(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "llmdebug_controller_packet.json").write_text(
            '{"schema_version": "v1"',
            encoding="utf-8",
        )
        (context_dir / "llmdebug_evidence_bundle.json").write_text(
            "[]",
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert "adaptive_controller_packet_loaded" not in result.meta
        assert result.meta["adaptive_controller_packet_error"].startswith("JSONDecodeError:")
        assert result.meta["adaptive_evidence_bundle_error"] == "invalid_sidecar_type:list"
        assert result.meta["baseline_hybrid_guidance_mode"] == "tuned_v1"
        assert result.meta["baseline_hybrid_guidance_chars"] == 0
        assert result.meta["baseline_hybrid_guidance_sections"] == []
        assert result.meta["baseline_case_conversion_sec"] >= 0.0
        assert result.meta["baseline_sidecar_load_sec"] >= 0.0
        assert adapter.seen_case_input is not None
        assert "llmdebug_controller_packet" not in adapter.seen_case_input
        assert "llmdebug_evidence" not in adapter.seen_case_input

    @pytest.mark.parametrize(
        ("mode", "expected_include_history", "expected_include_retry_plan", "expected_max_chars"),
        [
            ("fair_v1", True, True, None),
            ("fixed_v1", False, False, 4000),
        ],
    )
    def test_baseline_hybrid_guidance_metadata_intervenor_shared_modes_double_char_count(
        self,
        mode: str,
        expected_include_history: bool,
        expected_include_retry_plan: bool,
        expected_max_chars: int | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter(name="intervenor")
        config = BaselineConfig(
            extra={
                "baseline_hybrid_guidance_mode": mode,
                "baseline_intervenor_teacher_guidance_max_chars": 4000,
                "baseline_intervenor_learner_guidance_max_chars": 6000,
            }
        )
        observed: dict[str, Any] = {}

        class _RenderedGuidance:
            char_count = 17

        def _fake_render(
            controller_packet: Any,
            *,
            mode: str,
            include_history: bool,
            include_retry_plan: bool,
            max_chars: int | None = None,
        ) -> _RenderedGuidance:
            observed.update(
                {
                    "controller_packet": controller_packet,
                    "mode": mode,
                    "include_history": include_history,
                    "include_retry_plan": include_retry_plan,
                    "max_chars": max_chars,
                }
            )
            return _RenderedGuidance()

        monkeypatch.setattr(
            baseline_patcher_module, "render_hybrid_controller_guidance", _fake_render
        )
        monkeypatch.setattr(
            baseline_patcher_module,
            "hybrid_guidance_metadata",
            lambda rendered: {
                "baseline_hybrid_guidance_mode": mode,
                "baseline_hybrid_guidance_sections": ["controller_state"],
                "rendered_char_count": rendered.char_count,
            },
        )

        metadata = _baseline_hybrid_guidance_metadata(
            adapter=adapter,
            config=config,
            controller_packet_raw={"controller_state": {"helper_action_executed": "snapshot_lite"}},
        )

        assert metadata == {
            "baseline_hybrid_guidance_mode": mode,
            "baseline_hybrid_guidance_sections": ["controller_state"],
            "rendered_char_count": 17,
            "baseline_hybrid_guidance_chars": 34,
        }
        assert observed["mode"] == mode
        assert observed["include_history"] is expected_include_history
        assert observed["include_retry_plan"] is expected_include_retry_plan
        assert observed["max_chars"] == expected_max_chars

    def test_baseline_hybrid_guidance_metadata_intervenor_uses_rich_renderer_sections(
        self,
    ) -> None:
        adapter = _StubAdapter(name="intervenor")
        adapter._render_controller_guidance = lambda **kwargs: (  # type: ignore[attr-defined]
            "teacher_focus: isolate crash\n"
            if kwargs["focus"] == "teacher"
            else "learner_focus: keep the fix local\n"
        )
        config = BaselineConfig(
            extra={
                "baseline_intervenor_teacher_guidance_max_chars": 111,
                "baseline_intervenor_learner_guidance_max_chars": 222,
            }
        )

        metadata = _baseline_hybrid_guidance_metadata(
            adapter=adapter,
            config=config,
            controller_packet_raw={"controller_state": {"helper_action_executed": "frame_only"}},
        )

        assert metadata == {
            "baseline_hybrid_guidance_mode": "tuned_v1",
            "baseline_hybrid_guidance_chars": len(
                "teacher_focus: isolate crash\nlearner_focus: keep the fix local\n"
            ),
            "baseline_hybrid_guidance_sections": ["teacher_focus", "learner_focus"],
        }

    def test_baseline_hybrid_guidance_metadata_intervenor_without_renderer_returns_empty_sections(
        self,
    ) -> None:
        adapter = _StubAdapter(name="intervenor")
        config = BaselineConfig()

        metadata = _baseline_hybrid_guidance_metadata(
            adapter=adapter,
            config=config,
            controller_packet_raw={"controller_state": {"helper_action_executed": "search_only"}},
        )

        assert metadata == {
            "baseline_hybrid_guidance_mode": "tuned_v1",
            "baseline_hybrid_guidance_chars": 0,
            "baseline_hybrid_guidance_sections": [],
        }

    @pytest.mark.parametrize(
        ("adapter_name", "extra", "expected_max_chars"),
        [
            (
                "tracecoder",
                {"baseline_tracecoder_controller_guidance_max_chars": 321},
                321,
            ),
            (
                "lambda_rlm",
                {"baseline_lambda_controller_guidance_max_chars": 654},
                654,
            ),
            (
                "rlm",
                {"baseline_rlm_controller_guidance_max_chars": 987},
                987,
            ),
        ],
    )
    def test_baseline_hybrid_guidance_metadata_uses_adapter_specific_max_chars(
        self,
        adapter_name: str,
        extra: dict[str, int],
        expected_max_chars: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _StubAdapter(name=adapter_name)
        config = BaselineConfig(extra=extra)
        observed: dict[str, Any] = {}

        monkeypatch.setattr(
            baseline_patcher_module,
            "render_hybrid_controller_guidance",
            lambda controller_packet,
            *,
            mode,
            include_history,
            include_retry_plan,
            max_chars=None: (
                observed.update(
                    {
                        "controller_packet": controller_packet,
                        "mode": mode,
                        "include_history": include_history,
                        "include_retry_plan": include_retry_plan,
                        "max_chars": max_chars,
                    }
                )
                or "controller_state: inspect the crash"
            ),
        )
        monkeypatch.setattr(
            baseline_patcher_module,
            "hybrid_guidance_metadata",
            lambda rendered: {
                "baseline_hybrid_guidance_mode": "tuned_v1",
                "baseline_hybrid_guidance_chars": len(rendered),
                "baseline_hybrid_guidance_sections": ["controller_state"],
            },
        )

        metadata = _baseline_hybrid_guidance_metadata(
            adapter=adapter,
            config=config,
            controller_packet_raw={"controller_state": {"helper_action_executed": "snapshot_lite"}},
        )

        assert metadata == {
            "baseline_hybrid_guidance_mode": "tuned_v1",
            "baseline_hybrid_guidance_chars": len("controller_state: inspect the crash"),
            "baseline_hybrid_guidance_sections": ["controller_state"],
        }
        assert observed["mode"] == "tuned_v1"
        assert observed["include_history"] is True
        assert observed["include_retry_plan"] is True
        assert observed["max_chars"] == expected_max_chars

    def test_propose_patch_loads_controller_packet_without_sidecar_version(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "llmdebug_controller_packet.json").write_text(
            json.dumps(
                {"schema_version": "v1", "controller_state": {"helper_action_executed": "skip"}}
            ),
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["adaptive_controller_packet_loaded"] is True
        assert result.meta["adaptive_controller_schema_version"] == "v1"
        assert "adaptive_sidecar_schema_version" not in result.meta
        assert adapter.seen_case_input is not None
        assert adapter.seen_case_input["llmdebug_controller_packet"]["schema_version"] == "v1"

    def test_propose_patch_reports_corrupted_history_and_verifier_sidecars(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "llmdebug_history_bundle.json").write_text("[]", encoding="utf-8")
        (context_dir / "llmdebug_verifier_report.json").write_text(
            '{"schema_version": "v1"',
            encoding="utf-8",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["adaptive_history_bundle_error"] == "invalid_sidecar_type:list"
        assert result.meta["adaptive_verifier_report_error"].startswith("JSONDecodeError:")
        assert adapter.seen_case_input is not None
        assert "llmdebug_history" not in adapter.seen_case_input
        assert "llmdebug_verifier_report" not in adapter.seen_case_input

    def test_propose_patch_patch_diff_add_update_delete_round_trips(
        self,
        stub_case: Path,
        tmp_path: Path,
    ) -> None:
        custom_diff = (
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def stub():\n"
            "-    return 0\n"
            "+    return 42\n"
            "diff --git a/extra.py b/extra.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/extra.py\n"
            "@@ -0,0 +1 @@\n"
            "+EXTRA = True\n"
            "diff --git a/helper.py b/helper.py\n"
            "deleted file mode 100644\n"
            "--- a/helper.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-HELPER = True\n"
        )
        (stub_case / "helper.py").write_text("HELPER = True\n", encoding="utf-8")
        adapter = _StubAdapter(
            repair_result=BaselineResult(
                repaired_code="def stub():\n    return 41\n",
                patch_diff=custom_diff,
                success=True,
            )
        )
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.diff == custom_diff
        assert result.meta is not None
        assert result.meta["repair_scope"] == "patch_diff"

        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / "buggy.py").write_text("def stub():\n    return 0\n", encoding="utf-8")
        (workdir / "helper.py").write_text("HELPER = True\n", encoding="utf-8")
        ok, msg = apply_diff(workdir, result.diff or "")

        assert ok is True, msg
        assert (workdir / "buggy.py").read_text(encoding="utf-8") == "def stub():\n    return 42\n"
        assert (workdir / "extra.py").read_text(encoding="utf-8") == "EXTRA = True\n"
        assert not (workdir / "helper.py").exists()

    def test_propose_patch_falls_back_to_contract_version_when_adapter_contract_leaves_it_blank(
        self,
        stub_case: Path,
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

        adapter.contract = lambda: BaselineContract(
            baseline_name=adapter.name,
            contract_version="",
            repair_scope_policy="adapter_defined",
            prompt_contract_kind="adapter_defined",
            output_contract="adapter_defined",
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=stub_case, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["contract_version"] == BASELINE_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestBuildBaselinePatcher:
    """Tests for the build_baseline_patcher factory."""

    def test_baseline_config_normalizes_split_timeouts(self) -> None:
        config = BaselineConfig(timeout_sec=12)

        assert config.timeout_sec == 12
        assert config.process_timeout_sec == 12
        assert config.api_timeout_sec == 12

        explicit = BaselineConfig(api_timeout_sec=5, process_timeout_sec=9, timeout_sec=None)
        assert explicit.api_timeout_sec == 5
        assert explicit.process_timeout_sec == 9
        assert explicit.timeout_sec == 9

    def test_exact_baseline_char_totals_returns_none_for_missing_values(self) -> None:
        assert _exact_baseline_char_totals(None) is None
        assert _exact_baseline_char_totals({"prompt_chars_total": "12"}) is None

    def test_infer_repair_contract_returns_unknown_when_result_has_no_targets(self) -> None:
        scope, targets = _infer_repair_contract(BaselineResult(target_path=""))

        assert scope == "unknown"
        assert targets == []

    def test_infer_repair_contract_falls_back_from_blank_meta_targets_to_file_edits(self) -> None:
        scope, targets = _infer_repair_contract(
            BaselineResult(
                file_edits=[BaselineFileEdit(path="buggy.py", content="x = 1\n")],
                meta={"repair_targets": [" ", ""]},
            )
        )

        assert scope == "single_file"
        assert targets == ["buggy.py"]

    def test_infer_repair_contract_falls_back_from_blank_file_edit_targets_to_patch_diff(
        self,
    ) -> None:
        scope, targets = _infer_repair_contract(
            BaselineResult(
                file_edits=[BaselineFileEdit(path=" ", content="x = 1\n")],
                patch_diff="diff --git a/buggy.py b/buggy.py\n",
            )
        )

        assert scope == "patch_diff"
        assert targets == []

    def test_infer_repair_contract_prefers_file_edits_over_patch_diff(self) -> None:
        scope, targets = _infer_repair_contract(
            BaselineResult(
                file_edits=[BaselineFileEdit(path="helper.py", content="VALUE = 1\n")],
                patch_diff="diff --git a/buggy.py b/buggy.py\n",
                repaired_code="def stub():\n    return 42\n",
            )
        )

        assert scope == "single_file"
        assert targets == ["helper.py"]

    def test_unknown_baseline_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown baseline"):
            build_baseline_patcher(
                baseline_name="nonexistent",
                repo_dir=tmp_path,
            )

    def test_ldb_setup_fails_without_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            build_baseline_patcher(
                baseline_name="ldb",
                repo_dir=tmp_path / "missing",
            )

    def test_ldb_factory_succeeds_with_valid_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "ldb_repo"
        prog = repo / "programming"
        prog.mkdir(parents=True)
        (prog / "ldb.py").write_text(
            'dataset = "llmdebug"\n'
            "llm_duration_sec = 0.0\n"
            "usage_available = True\n"
            "is_passing = False\n"
            "test_feedback = []\n",
            encoding="utf-8",
        )
        (prog / "main.py").write_text("# stub", encoding="utf-8")
        (prog / "utils.py").write_text('dataset = "llmdebug"\n', encoding="utf-8")
        generators = prog / "generators"
        generators.mkdir()
        (generators / "model.py").write_text(
            "_record_metrics = True\n"
            "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE = 'auto'\n"
            "LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL = 'test-model'\n"
            "compat_error = None\n"
            "super().__init__(model_name)\n",
            encoding="utf-8",
        )
        (generators / "py_generate.py").write_text(
            'dataset_type in ["HumanEval", "MBPP", "llmdebug"]\n',
            encoding="utf-8",
        )

        patcher = build_baseline_patcher(
            baseline_name="ldb",
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
        )
        assert patcher.name == "ldb"
        assert patcher.config.model == "test-model"
        assert patcher.config.max_attempts == 2

    def test_tracecoder_factory_succeeds_with_valid_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "tracecoder_repo"
        (repo / "src").mkdir(parents=True)
        (repo / "datasets" / "llmdebug").mkdir(parents=True)
        (repo / "config.py").write_text(
            'DATASET_PATHS = {"llmdebug": {"data_path": "x", "eval_module": "datasets.llmdebug.evaluation"}}\n'
            "# --dataset-path support\n",
            encoding="utf-8",
        )
        (repo / "problem_processor.py").write_text(
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
            "exact_test_context_meta\n",
            encoding="utf-8",
        )
        (repo / "trace_learn_coder.py").write_text(
            "llmdebug_metrics_path = ''\n",
            encoding="utf-8",
        )
        (repo / "src" / "generation.py").write_text(
            "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE = 'auto'\n"
            "openai_compat_reasoning_only = False\n"
            "TRACECODER_ANALYSIS_MAX_TOKENS = 2048\n"
            "TRACECODER_IMPLEMENTATION_MAX_TOKENS = 4096\n",
            encoding="utf-8",
        )
        (repo / "reporting.py").write_text(
            "build_llmdebug_failure_feedback = True\nsave_metrics_sidecar = True\n",
            encoding="utf-8",
        )
        (repo / "src" / "exact_test_context.py").write_text(
            "def build_exact_test_context(*args, **kwargs):\n    return '', {}\n",
            encoding="utf-8",
        )
        (repo / "src" / "traceRunner.py").write_text("# stub\n", encoding="utf-8")
        (repo / "datasets" / "llmdebug" / "evaluation.py").write_text(
            "def check_correctness(*args, **kwargs):\n    return {}\n",
            encoding="utf-8",
        )

        patcher = build_baseline_patcher(
            baseline_name="tracecoder",
            repo_dir=repo,
            model="test-model",
            max_attempts=2,
        )
        assert patcher.name == "tracecoder"
        assert patcher.config.model == "test-model"
        assert patcher.config.max_attempts == 2

    def test_factory_accepts_split_timeout_fields(self, tmp_path: Path) -> None:
        repo = tmp_path / "tracecoder_repo"
        (repo / "src").mkdir(parents=True)
        (repo / "datasets" / "llmdebug").mkdir(parents=True)
        (repo / "config.py").write_text(
            'DATASET_PATHS = {"llmdebug": {"data_path": "x", "eval_module": "datasets.llmdebug.evaluation"}}\n'
            "# --dataset-path support\n",
            encoding="utf-8",
        )
        (repo / "problem_processor.py").write_text(
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
            "exact_test_context_meta\n",
            encoding="utf-8",
        )
        (repo / "trace_learn_coder.py").write_text(
            "llmdebug_metrics_path = ''\n",
            encoding="utf-8",
        )
        (repo / "src" / "generation.py").write_text(
            "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE = 'auto'\n"
            "openai_compat_reasoning_only = False\n"
            "TRACECODER_ANALYSIS_MAX_TOKENS = 2048\n"
            "TRACECODER_IMPLEMENTATION_MAX_TOKENS = 4096\n",
            encoding="utf-8",
        )
        (repo / "reporting.py").write_text(
            "build_llmdebug_failure_feedback = True\nsave_metrics_sidecar = True\n",
            encoding="utf-8",
        )
        (repo / "src" / "exact_test_context.py").write_text(
            "def build_exact_test_context(*args, **kwargs):\n    return '', {}\n",
            encoding="utf-8",
        )
        (repo / "src" / "traceRunner.py").write_text("# stub\n", encoding="utf-8")
        (repo / "datasets" / "llmdebug" / "evaluation.py").write_text(
            "def check_correctness(*args, **kwargs):\n    return {}\n",
            encoding="utf-8",
        )

        patcher = build_baseline_patcher(
            baseline_name="tracecoder",
            repo_dir=repo,
            api_timeout_sec=13,
            process_timeout_sec=27,
        )

        assert patcher.config.api_timeout_sec == 13
        assert patcher.config.process_timeout_sec == 27
        assert patcher.config.timeout_sec == 27

    def test_max_attempts_is_independent_of_eval_k(self, tmp_path: Path) -> None:
        """max_attempts controls adapter-internal iterations, not eval-level k."""
        repo = tmp_path / "intervenor_repo"
        repo.mkdir()
        (repo / "intervenor").mkdir()
        (repo / "intervenor" / "__init__.py").write_text("", encoding="utf-8")
        (repo / "intervenor" / "teacher.py").write_text("# stub\n", encoding="utf-8")
        (repo / "intervenor" / "learner.py").write_text("# stub\n", encoding="utf-8")
        prompt_dir = repo / "prompt"
        prompt_dir.mkdir()
        (prompt_dir / "code_teacher.txt").write_text("# stub\n", encoding="utf-8")
        (prompt_dir / "code_learner.txt").write_text("# stub\n", encoding="utf-8")

        patcher = build_baseline_patcher(
            baseline_name="intervenor",
            repo_dir=repo,
            max_attempts=1,
        )
        assert patcher.config.max_attempts == 1

        patcher2 = build_baseline_patcher(
            baseline_name="intervenor",
            repo_dir=repo,
            max_attempts=5,
        )
        assert patcher2.config.max_attempts == 5


class _PromptStubAdapter(_StubAdapter):
    retry_context_prompt_field = "prompt"

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> dict[str, Any]:
        return {**super().convert_case(case_dir, case_context=case_context), "prompt": "Fix it."}


class TestRetryContext:  # BAS-02 / D4
    def test_retry_attempt_appends_failure_to_prompt_field(
        self, stub_case: Path, tmp_path: Path
    ) -> None:
        adapter = _PromptStubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())
        context_dir = tmp_path / "ctx" / "attempt_02"
        context_dir.mkdir(parents=True)
        (context_dir / "pytest_before_combined.txt").write_text(
            "E   AssertionError: assert 0 == 42\n", encoding="utf-8"
        )

        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=stub_case)

        assert adapter.seen_case_input is not None
        assert "assert 0 == 42" in adapter.seen_case_input["prompt"]
        assert adapter.seen_case_input["llmdebug_retry_context"]["attempt"] == 2
        assert result.meta is not None
        assert result.meta["baseline_retry_context_consumed"] is True

    def test_first_attempt_and_adapters_without_prompt_field(
        self, stub_case: Path, tmp_path: Path
    ) -> None:
        adapter = _StubAdapter()
        patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())
        first = tmp_path / "ctx" / "attempt_01"
        first.mkdir(parents=True)
        result = patcher.propose_patch(prompt="ignored", context_dir=first, cwd=stub_case)
        assert result.meta is not None
        assert result.meta["baseline_retry_context_consumed"] is False

        second = tmp_path / "ctx" / "attempt_02"
        second.mkdir(parents=True)
        result = patcher.propose_patch(prompt="ignored", context_dir=second, cwd=stub_case)
        assert result.meta is not None
        assert result.meta["baseline_retry_context_consumed"] is False  # no prompt field


class _RecordingStubAdapter(_StubAdapter):
    """Reports repo state and case information like the real adapters (BAS-14, ATT-02)."""

    repo_state = {"observed_commit": "abc", "dirty": False, "tree_patch_sha256": "f00"}

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> dict[str, Any]:
        converted = super().convert_case(case_dir, case_context=case_context)
        return {**converted, "case_information": {"case_info_policy": "neutral_v1"}}


class TestRecordedBaselineMetadata:
    def test_repo_state_and_case_information_reach_patch_meta(
        self, stub_case: Path, tmp_path: Path
    ) -> None:
        patcher = BaselinePatcher(adapter=_RecordingStubAdapter(), config=BaselineConfig())

        result = patcher.propose_patch(prompt="ignored", context_dir=tmp_path, cwd=stub_case)

        assert result.meta is not None
        assert result.meta["baseline_repo_state"] == _RecordingStubAdapter.repo_state
        assert result.meta["case_information"] == {"case_info_policy": "neutral_v1"}


class _BrokenConverterAdapter(_StubAdapter):
    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> dict[str, Any]:
        if case_dir.name == "incompatible":
            raise BaselineCaseIncompatibleError(
                reason="subprocess_task_unsupported", detail="no entry point"
            )
        if case_dir.name == "broken":
            raise KeyError("converter bug")
        return super().convert_case(case_dir, case_context=case_context)


def test_baseline_case_support_counts_converter_errors_separately(tmp_path: Path) -> None:
    """BAS-01: a converter bug is counted under its own reason, not hidden."""
    case_dirs: list[tuple[str, Path]] = []
    for name in ("ok", "incompatible", "broken"):
        case_dir = tmp_path / name
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text("x = 1\n", encoding="utf-8")
        case_dirs.append((name, case_dir))

    support = baseline_patcher_module.baseline_case_support(_BrokenConverterAdapter(), case_dirs)

    assert support["convertible_cases"] == 1
    assert support["incompatible_reasons"] == {
        "conversion_error:KeyError": 1,
        "subprocess_task_unsupported": 1,
    }
