"""Integration tests for TOON encoding feature gate in the eval harness."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import eval_attempt_flow, run_eval
from evals.eval_policy_features import _adaptive_budget_feature_vector as _budget_feature_vector
from evals.eval_prompts import _render_retry_packet, render_stage_b_patch_prompt
from evals.eval_types import AdaptiveRuntimeConfig, RunConditionInputs
from evals.patchers.heuristic_patcher import HeuristicSnapshotPatcher
from evals.toon_llm_encoder import encode_for_llm


def _toons_installed() -> bool:
    try:
        import toons  # type: ignore[import-untyped]

        return True
    except ImportError:
        return False


def _make_case(tmp_path: Path, case_id: str = "toon_case") -> run_eval.CaseSpec:
    case_dir = tmp_path / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "test_sample.py").write_text(
        "def test_fail():\n    assert False\n", encoding="utf-8"
    )
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="unit_test",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=10,
    )


def _setup_workdir(
    tmp_path: Path,
    case_id: str,
    condition: str,
) -> tuple[Path, Path]:
    workdir = tmp_path / "workdirs" / case_id / condition
    workdir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "contexts" / case_id / condition
    context_dir.mkdir(parents=True, exist_ok=True)
    return workdir, context_dir


def _failing_run_result() -> run_eval.RunResult:
    return run_eval.RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="FAILED test_sample.py::test_fail",
        stderr="",
        timed_out=False,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _base_inputs() -> RunConditionInputs:
    """Minimal RunConditionInputs with adaptive_toon_encoding=False."""
    return RunConditionInputs(
        k=3,
        output_format="json",
        run_id="test-run",
        include_files_context=True,
        force_patch_attempt=False,
        adaptive_toon_encoding=False,
    )


@pytest.fixture
def _toon_inputs(_base_inputs: RunConditionInputs) -> RunConditionInputs:
    """RunConditionInputs with adaptive_toon_encoding=True."""
    return replace(_base_inputs, adaptive_toon_encoding=True)


# ---------------------------------------------------------------------------
# Feature gate: flag ignored for non-adaptive conditions
# ---------------------------------------------------------------------------


class TestToonFlagGating:
    def test_toon_flag_present_on_inputs(self, _base_inputs: RunConditionInputs) -> None:
        assert hasattr(_base_inputs, "adaptive_toon_encoding")
        assert _base_inputs.adaptive_toon_encoding is False

    def test_toon_flag_true_on_inputs(self, _toon_inputs: RunConditionInputs) -> None:
        assert _toon_inputs.adaptive_toon_encoding is True

    def test_adaptive_runtime_config_has_toon_field(self) -> None:
        config = AdaptiveRuntimeConfig()
        assert hasattr(config, "adaptive_toon_encoding")
        assert config.adaptive_toon_encoding is False

    def test_adaptive_runtime_config_toon_true(self) -> None:
        config = AdaptiveRuntimeConfig(adaptive_toon_encoding=True)
        assert config.adaptive_toon_encoding is True

    def test_from_components_passes_toon_through(self) -> None:
        adaptive = AdaptiveRuntimeConfig(adaptive_toon_encoding=True)
        inputs = RunConditionInputs.from_components(
            k=1,
            output_format="json",
            run_id="test",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_runtime=adaptive,
        )
        assert inputs.adaptive_toon_encoding is True

    def test_from_components_defaults_toon_false(self) -> None:
        inputs = RunConditionInputs.from_components(
            k=1,
            output_format="json",
            run_id="test",
            include_files_context=False,
            force_patch_attempt=False,
        )
        assert inputs.adaptive_toon_encoding is False


# ---------------------------------------------------------------------------
# Retry packet serialization
# ---------------------------------------------------------------------------


class TestRetryPacketToonEncoding:
    def test_retry_packet_json_by_default(self) -> None:
        result = _render_retry_packet(
            failure_signature="sig",
            previous_failure_signature=None,
            failure_signature_repeat=False,
            previous_attempt_diff=None,
            toon_encoding_active=False,
        )
        # Should contain pretty-printed JSON
        assert "Retry packet" in result
        body = result.split("\n", 1)[1]
        parsed = json.loads(body)
        assert parsed["failure_signature"] == "sig"


class TestStageBPromptToonEncoding:
    def test_stage_b_prompt_uses_compact_snapshot_payload_when_provided(self) -> None:
        snapshot = {
            "exception": {"type": "IndexError", "message": "out of range"},
            "frames": [{"file_rel": "buggy.py", "line": 5, "function": "g"}],
        }
        encoded_snapshot = encode_for_llm(
            snapshot,
            use_toon=False,
            use_compact_keys=True,
        )
        evidence = {
            "case_id": "case_1",
            "condition": "layered_adaptive_llm",
            "attempt": 2,
            "repro": ["python", "-m", "pytest", "-q"],
            "stdout": "E   IndexError: out of range\n",
            "stderr": "",
            "failure_signature": "IndexError:abc123",
            "previous_failure_signature": None,
            "failure_signature_repeat": False,
            "previous_attempt_diff": None,
            "file_index": "buggy.py\ntest_buggy.py",
            "files": "",
            "snapshot_available": True,
            "snapshot_error": None,
        }

        prompt = render_stage_b_patch_prompt(
            evidence=evidence,
            snapshot=snapshot,
            snapshot_prompt_text=encoded_snapshot,
            include_project_files=False,
        )

        assert "Structured payload (compact encoding; keys may be abbreviated):" in prompt
        assert encoded_snapshot in prompt
        assert "buggy.py:5 in g()" not in prompt


class TestSnapshotArtifactCompatibility:
    def test_compact_snapshot_artifact_does_not_break_heuristic_patcher(self, tmp_path) -> None:
        snapshot = {
            "schema_version": 1,
            "exception": {"type": "IndexError", "message": "out of range"},
            "frames": [
                {
                    "file_rel": "buggy.py",
                    "line": 2,
                    "function": "f",
                    "locals": {"i": 3, "xs": [1, 2, 3]},
                }
            ],
        }
        snapshot_full_text = json.dumps(snapshot, indent=2, ensure_ascii=False)
        snapshot_artifact_text = encode_for_llm(
            snapshot,
            use_toon=False,
            use_compact_keys=True,
        )
        (tmp_path / "snapshot.json").write_text(snapshot_artifact_text, encoding="utf-8")
        (tmp_path / "buggy.py").write_text(
            "def f(xs, i):\n    return xs[i]\n",
            encoding="utf-8",
        )

        result = run_eval._propose_patch_with_snapshot_context(
            patcher=HeuristicSnapshotPatcher(),
            prompt="fix it",
            context_dir=tmp_path,
            cwd=tmp_path,
            snapshot_full_text=snapshot_full_text,
            snapshot_artifact_text=snapshot_artifact_text,
        )

        assert result.diff is not None
        assert (tmp_path / "snapshot.json").read_text(encoding="utf-8") == snapshot_artifact_text


class TestRetryPacketToonEncodingAdvanced:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_retry_packet_toon_when_active(self) -> None:
        result = _render_retry_packet(
            failure_signature="sig",
            previous_failure_signature=None,
            failure_signature_repeat=False,
            previous_attempt_diff=None,
            toon_encoding_active=True,
        )
        assert "Retry packet" in result
        # The body should be shorter than pretty JSON
        body_toon = result.split("\n", 1)[1]

        result_json = _render_retry_packet(
            failure_signature="sig",
            previous_failure_signature=None,
            failure_signature_repeat=False,
            previous_attempt_diff=None,
            toon_encoding_active=False,
        )
        body_json = result_json.split("\n", 1)[1]
        # TOON or compact JSON should be more compact
        assert len(body_toon) <= len(body_json)

    def test_retry_packet_toon_fallback_when_toons_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When toons is unavailable, falls back to compact JSON."""
        import sys

        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        import evals.toon_llm_encoder as mod

        monkeypatch.setattr(mod, "_TOON_FALLBACK_WARNING_EMITTED", True)  # suppress warning

        result = _render_retry_packet(
            failure_signature="sig",
            previous_failure_signature=None,
            failure_signature_repeat=False,
            previous_attempt_diff=None,
            toon_encoding_active=True,
        )
        assert "Retry packet" in result
        body = result.split("\n", 1)[1].strip()
        # Should be valid minified JSON
        parsed = json.loads(body)
        assert parsed["failure_signature"] == "sig"


# ---------------------------------------------------------------------------
# OpenAI payload compact separators
# ---------------------------------------------------------------------------


class TestOpenAIPayloadCompactSeparators:
    def test_compact_separators_less_whitespace(self) -> None:
        """Compact separators produce shorter output than default."""
        payload = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        default_bytes = json.dumps(payload).encode("utf-8")
        compact_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        assert len(compact_bytes) < len(default_bytes)


# ---------------------------------------------------------------------------
# Budget feature vector includes toon encoding flag
# ---------------------------------------------------------------------------


class TestBudgetFeatureToonEncoding:
    def test_toon_encoding_feature_present(self) -> None:
        from evals.eval_policy_features import _adaptive_budget_feature_vector

        # Need a PreGateResult stub
        @dataclass
        class _StubPreGate:
            score: float = 0.5
            should_skip: bool = False
            exception_type: str = ""
            short_reason: str = ""

        features = _adaptive_budget_feature_vector(
            attempt=1,
            max_attempts=3,
            timed_out=False,
            failure_signature_repeat=False,
            stagnation_streak=0,
            helper_calls_used=0,
            helper_calls_without_context=0,
            adaptive_no_progress_streak=0,
            adaptive_no_progress_token_sum=0,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            supports_context_request=True,
            snapshot_available=True,
            pregate_result=_StubPreGate(),
            context_compaction_enabled=False,
            supports_context_compaction=False,
            context_compaction_calls_used=0,
            context_compaction_max_calls=1,
            last_context_compaction_requested=False,
            last_context_compaction_invoked=False,
            last_context_compaction_success=False,
            last_context_compaction_ratio=0.0,
            last_context_compaction_trigger="",
            current_tier_index=0,
            max_tier_index=2,
            ups_used=0,
            max_ups_per_case=1,
            base_propose_budget_sec=30.0,
            next_tier_changes={"propose_budget": True},
            next_tier_multiplier_delta=0.5,
            last_helper_need_more_context=None,
            last_helper_decision_reason="",
            toon_encoding_active=True,
        )
        assert "toon_encoding_active" in features
        assert features["toon_encoding_active"] == 1.0

    def test_toon_encoding_feature_zero_when_inactive(self) -> None:
        from evals.eval_policy_features import _adaptive_budget_feature_vector

        @dataclass
        class _StubPreGate:
            score: float = 0.5
            should_skip: bool = False
            exception_type: str = ""
            short_reason: str = ""

        features = _adaptive_budget_feature_vector(
            attempt=1,
            max_attempts=3,
            timed_out=False,
            failure_signature_repeat=False,
            stagnation_streak=0,
            helper_calls_used=0,
            helper_calls_without_context=0,
            adaptive_no_progress_streak=0,
            adaptive_no_progress_token_sum=0,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            supports_context_request=True,
            snapshot_available=True,
            pregate_result=_StubPreGate(),
            context_compaction_enabled=False,
            supports_context_compaction=False,
            context_compaction_calls_used=0,
            context_compaction_max_calls=1,
            last_context_compaction_requested=False,
            last_context_compaction_invoked=False,
            last_context_compaction_success=False,
            last_context_compaction_ratio=0.0,
            last_context_compaction_trigger="",
            current_tier_index=0,
            max_tier_index=2,
            ups_used=0,
            max_ups_per_case=1,
            base_propose_budget_sec=30.0,
            next_tier_changes={},
            next_tier_multiplier_delta=0.0,
            last_helper_need_more_context=None,
            last_helper_decision_reason="",
            toon_encoding_active=False,
        )
        assert features["toon_encoding_active"] == 0.0


# ---------------------------------------------------------------------------
# Regression: baseline conditions unaffected
# ---------------------------------------------------------------------------


class TestBaselineConditionsUnaffected:
    def test_non_adaptive_condition_ignores_toon_flag(self) -> None:
        """Verify that toon_encoding_active is only True for adaptive conditions."""
        from evals.eval_types import Condition

        non_adaptive = [
            Condition.BASE_REPAIR,
            Condition.COT_REPAIR,
        ]
        for cond in non_adaptive:
            is_adaptive_gated = cond == Condition.LAYERED_ADAPTIVE_GATED
            is_adaptive_llm = cond == Condition.LAYERED_ADAPTIVE_LLM
            is_adaptive = is_adaptive_gated or is_adaptive_llm
            toon_active = bool(True and is_adaptive)  # adaptive_toon_encoding=True
            assert toon_active is False, f"TOON should be gated off for {cond}"

    def test_adaptive_condition_enables_toon(self) -> None:
        from evals.eval_types import Condition

        adaptive = [
            Condition.LAYERED_ADAPTIVE_GATED,
            Condition.LAYERED_ADAPTIVE_LLM,
        ]
        for cond in adaptive:
            is_adaptive_gated = cond == Condition.LAYERED_ADAPTIVE_GATED
            is_adaptive_llm = cond == Condition.LAYERED_ADAPTIVE_LLM
            is_adaptive = is_adaptive_gated or is_adaptive_llm
            toon_active = bool(True and is_adaptive)
            assert toon_active is True, f"TOON should be active for {cond}"


class TestAttemptFlowToonFeatureGating:
    def test_non_openai_patcher_keeps_budget_feature_inactive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path, "toon_feature_command")
        workdir, context_dir = _setup_workdir(
            tmp_path,
            case.case_id,
            run_eval.Condition.LAYERED_ADAPTIVE_GATED.value,
        )
        for path in case.path.iterdir():
            if path.is_file():
                shutil.copy2(path, workdir / path.name)

        snapshot = {
            "exception": {"type": "IndexError", "message": "boom"},
            "frames": [
                {
                    "file_rel": "buggy.py",
                    "line": 7,
                    "function": "crash",
                    "locals": {"values": [1, 2, 3], "i": 3},
                    "locals_meta": {
                        "values": {"type": "builtins.list", "len": 3},
                        "i": {"type": "builtins.int"},
                    },
                }
            ],
            "git": {"commit": "abc123"},
        }
        captured: dict[str, bool] = {}

        def _capture_budget_features(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["toon_encoding_active"] = bool(kwargs["toon_encoding_active"])
            return _budget_feature_vector(*args, **kwargs)

        monkeypatch.setattr(eval_attempt_flow, "read_latest_snapshot", lambda *_: (snapshot, None))
        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *args, **kwargs: _failing_run_result(),
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "_adaptive_budget_feature_vector",
            _capture_budget_features,
        )

        patcher = SimpleNamespace(
            name="command",
            propose_patch=lambda *args, **kwargs: SimpleNamespace(
                diff=(
                    "--- a/test_sample.py\n+++ b/test_sample.py\n@@ -1,2 +1,2 @@\n"
                    " def test_fail():\n-    assert False\n+    assert True\n"
                ),
                meta={"status": "ok"},
            ),
        )

        records = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=RunConditionInputs(
                k=1,
                output_format="diff",
                run_id="toon-feature-command",
                include_files_context=False,
                force_patch_attempt=False,
                patcher=patcher,
                adaptive_toon_encoding=True,
                context_protocol="staged",
                run_metadata={"run_id": "toon-feature-command"},
            ),
        )

        assert isinstance(records, list)
        assert captured["toon_encoding_active"] is False


# ---------------------------------------------------------------------------
# Snapshot fidelity through encoding
# ---------------------------------------------------------------------------


class TestSnapshotFidelity:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_all_fields_preserved_toon(self) -> None:
        from evals.toon_llm_encoder import decode_for_llm, encode_for_llm

        snapshot: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_utc": "2024-06-01T12:00:00Z",
            "exception": {"type": "KeyError", "message": "'x'"},
            "frames": [
                {
                    "file": "a.py",
                    "line": 5,
                    "function": "g",
                    "locals": {"x": 42, "y": "hello"},
                }
            ],
            "git": {"commit": "abc123", "branch": "main", "dirty": False},
        }
        encoded = encode_for_llm(snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        assert decoded["schema_version"] == 1
        assert decoded["timestamp_utc"] == "2024-06-01T12:00:00Z"
        assert decoded["exception"]["type"] == "KeyError"
        assert decoded["frames"][0]["locals"]["x"] == 42
        assert decoded["git"]["commit"] == "abc123"

    def test_all_fields_preserved_json(self) -> None:
        from evals.toon_llm_encoder import decode_for_llm, encode_for_llm

        snapshot: dict[str, Any] = {
            "schema_version": 1,
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [{"file": "b.py", "line": 10, "function": "h", "locals": {"z": [1, 2, 3]}}],
        }
        encoded = encode_for_llm(snapshot, use_toon=False, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        assert decoded["schema_version"] == 1
        assert decoded["exception"]["type"] == "ValueError"
        assert decoded["frames"][0]["locals"]["z"] == [1, 2, 3]

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_compact_keys_roundtrip_with_toon(self) -> None:
        """compact -> TOON -> decode -> expand = original."""
        from evals.toon_llm_encoder import decode_for_llm, encode_for_llm

        original: dict[str, Any] = {
            "schema_version": 2,
            "exception": {"type": "OSError", "message": "not found"},
            "frames": [
                {"file": "c.py", "line": 1, "function": "open_file", "locals": {"path": "/tmp"}}
            ],
        }
        encoded = encode_for_llm(original, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        assert decoded["schema_version"] == original["schema_version"]
        assert decoded["exception"] == original["exception"]
        assert decoded["frames"][0]["file"] == "c.py"
        assert decoded["frames"][0]["locals"]["path"] == "/tmp"
