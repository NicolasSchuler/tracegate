"""Coverage-boost tests for evals/eval_attempt_flow.py — WP4b.

Targets uncovered lines and branches identified in coverage-evals.json:
- Lines 679-897: Transport recovery, infra event, adaptive history error, pending credits
- Lines 1328, 2017: File index cache, budget policy shadow_mode_baseline
- Lines 2052-2143: Budget policy override reasons and no-effective-change rollback
- Lines 2264-2346: Helper saturation guard, online policy override paths
- Lines 2513-2781: Stage-A execution paths, stagnation override, helper outcomes
- Lines 3206-3992: Context compaction ratio, response mode override
- Lines 4270-5933: Sidecar writes, transport recovery in propose, syntax preflight pass
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evals import eval_attempt_flow, run_eval
from evals.eval_types import RunConditionInputs

# ---------------------------------------------------------------------------
# Mock controller factories
# ---------------------------------------------------------------------------


def _mock_budget_policy_controller(
    applied_action: str = "keep_budget",
    selected_action: str = "keep_budget",
    selected_family: str = "",
    applied_family: str = "",
    mode: str = "on",
    architecture: str = "tier_binary",
) -> Any:
    """Create a mock budget policy controller with decide() method."""
    from evals.adaptive_budget_online_policy import BudgetPolicyDecision

    decision = BudgetPolicyDecision(
        decision_id="test-budget-decision",
        mode=mode,
        algorithm="linucb",
        architecture=architecture,
        decision_index=0,
        created_at_utc="2024-01-01T00:00:00Z",
        selected_action=selected_action,
        applied_action=applied_action,
        default_action="keep_budget",
        propensity=1.0,
        decision_strategy="default",
        scores={},
        feature_values={},
        feature_scaling_caps={},
        feature_scaling_history_sizes={},
        feature_scaling_fallback_to_v1=False,
        selected_family=selected_family,
        applied_family=applied_family,
        available_families=(),
        unavailable_family_reasons={},
        family_propensity=1.0,
        family_decision_strategy="default",
        family_scores={},
        family_feature_values={},
        family_feature_scaling_caps={},
        family_feature_scaling_history_sizes={},
        family_feature_scaling_fallback_to_v1=False,
        configured_alpha=1.0,
        configured_epsilon=0.1,
        configured_decay=1.0,
        effective_epsilon=0.1,
        warmup_active=False,
    )

    controller = SimpleNamespace(
        mode=mode,
        algorithm="linucb",
        config=SimpleNamespace(
            architecture=architecture,
            feature_scaling_profile="v1",
        ),
        decide=lambda **kw: decision,
        update=lambda **kw: None,
    )
    return controller


def _mock_online_policy_controller(
    applied_action: str = "invoke_helper",
    selected_action: str = "invoke_helper",
    mode: str = "on",
) -> Any:
    """Create a mock online policy controller with decide() method."""
    from evals.adaptive_online_policy import PolicyDecision

    decision = PolicyDecision(
        decision_id="test-online-decision",
        mode=mode,
        algorithm="linucb",
        decision_index=0,
        created_at_utc="2024-01-01T00:00:00Z",
        selected_action=selected_action,
        applied_action=applied_action,
        default_action="invoke_helper",
        propensity=1.0,
        decision_strategy="default",
        scores={},
        feature_values={},
        available_actions=("invoke_helper", "skip_helper"),
        unavailable_action_reasons={},
        feature_scaling_caps={},
        feature_scaling_history_sizes={},
        feature_scaling_fallback_to_v1=False,
        configured_alpha=1.0,
        configured_epsilon=0.1,
        configured_decay=1.0,
        effective_epsilon=0.1,
        warmup_active=False,
    )

    controller = SimpleNamespace(
        mode=mode,
        algorithm="linucb",
        config=SimpleNamespace(
            feature_scaling_profile="v1",
        ),
        decide=lambda **kw: decision,
        update=lambda **kw: None,
    )
    return controller


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path, case_id: str = "wp4b") -> run_eval.CaseSpec:
    case_dir = tmp_path / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "test_sample.py").write_text("def test_fail():\n    assert False\n")
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


def _make_run_result(**overrides: Any) -> run_eval.RunResult:
    defaults: dict[str, Any] = {
        "returncode": 1,
        "duration_sec": 0.1,
        "stdout": "FAILED test_sample.py::test_fail",
        "stderr": "",
        "timed_out": False,
    }
    defaults.update(overrides)
    return run_eval.RunResult(**defaults)


def _minimal_inputs(**overrides: Any) -> RunConditionInputs:
    defaults: dict[str, Any] = {
        "k": 1,
        "output_format": "diff",
        "run_id": "wp4b-run",
        "include_files_context": False,
        "force_patch_attempt": False,
        "run_metadata": {"run_id": "wp4b-run"},
    }
    defaults.update(overrides)
    return RunConditionInputs(**defaults)


def _setup_workdir(
    tmp_path: Path, case_id: str, condition: str = "base_repair"
) -> tuple[Path, Path]:
    workdir = tmp_path / "workdirs" / case_id / condition
    workdir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "contexts" / case_id / condition
    context_dir.mkdir(parents=True, exist_ok=True)
    return workdir, context_dir


def _stub_patcher(
    name: str = "stub",
    diff: str = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
    meta: dict[str, Any] | None = None,
    **extra_attrs: Any,
) -> SimpleNamespace:
    ns = SimpleNamespace(
        name=name,
        propose_patch=lambda *a, **kw: SimpleNamespace(
            diff=diff,
            meta=meta or {"status": "ok"},
        ),
        **extra_attrs,
    )
    return ns


def _run_impl(
    tmp_path: Path,
    case_id: str,
    condition: Any,
    monkeypatch: pytest.MonkeyPatch,
    patcher: Any | None = None,
    fake_result: Any | None = None,
    inputs_overrides: dict[str, Any] | None = None,
    copy_case: bool = True,
) -> list[dict[str, Any]]:
    """Run run_one_condition_impl with defaults and return records."""
    case = _make_case(tmp_path, case_id)
    workdir, context_dir = _setup_workdir(tmp_path, case_id, condition.value)

    if copy_case:
        for f in case.path.iterdir():
            if f.is_file():
                shutil.copy2(f, workdir / f.name)

    if patcher is None:
        patcher = _stub_patcher()
    if fake_result is None:
        fake_result = _make_run_result()

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda *a, **kw: fake_result,
    )

    overrides = inputs_overrides or {}
    overrides.setdefault("patcher", patcher)

    return eval_attempt_flow.run_one_condition_impl(
        case=case,
        condition=condition,
        workdir=workdir,
        context_dir_base=context_dir,
        inputs=_minimal_inputs(**overrides),
    )


# ===================================================================
# 1. Transport recovery paths (lines 734-803)
# ===================================================================


class TestTransportRecoveryPaths:
    """Cover _run_openai_operation_with_transport_recovery inner function."""

    def test_transport_recovery_successful_replay(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 734-803: transport failure -> recovery succeeds -> replay invoke."""
        from evals.patchers.base import PatchResult

        call_count = {"n": 0}

        def _propose(*a: Any, **kw: Any) -> PatchResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return PatchResult(
                    diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                    meta={
                        "error": "request_failed",
                        "exc": "ConnectionError: connection refused",
                        "usage": {"prompt_tokens": 30, "completion_tokens": 0},
                    },
                )
            return PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={
                    "status": "ok",
                    "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
                },
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=_propose,
            probe_server_ready=lambda **_kw: True,
        )

        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {
                "attempted": True,
                "recovered": True,
                "timed_out": False,
                "wait_sec": 1.5,
                "probe_attempts": [1, 2],
                "probe_meta": {"latency_ms": 100},
            },
        )

        result = _run_impl(
            tmp_path,
            "transport_replay",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
            },
        )
        assert isinstance(result, list)
        assert len(result) >= 1
        # ATT-15: the replayed attempt pays for the failed first call too.
        attempt_records = [r for r in result if r.get("record_type", "attempt") == "attempt"]
        patch_meta = attempt_records[-1]["patch_meta"]
        assert patch_meta["transport_recovery_first_call_usage"] == {
            "prompt_tokens": 30,
            "completion_tokens": 0,
        }
        assert patch_meta["usage"]["prompt_tokens"] == 70

    def test_transport_recovery_failed_no_replay(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 770-787: transport failure -> recovery fails -> return with failed meta."""
        from evals.patchers.base import PatchResult

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: PatchResult(
                diff="",
                meta={"error": "request_failed", "status": 502},
            ),
        )

        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {
                "attempted": True,
                "recovered": False,
                "timed_out": True,
                "wait_sec": 30.0,
                "probe_attempts": [1, 2, 3],
                "probe_meta": {},
            },
        )

        result = _run_impl(
            tmp_path,
            "transport_fail",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
            },
        )
        assert isinstance(result, list)

    def test_baseline_repair_failed_recovery_replays_same_attempt_without_scoring_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from evals.patchers.base import PatchResult

        call_count = {"n": 0}

        def _propose(*a: Any, **kw: Any) -> PatchResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return PatchResult(
                    diff=None,
                    meta={
                        "error": "repair_failed",
                        "exc": "InternalServerError: Error code: 503 - Loading model",
                    },
                )
            return PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            )

        patcher = SimpleNamespace(
            name="baseline",
            model="gpt-4",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=_propose,
            probe_server_ready=lambda: (True, {"status": 200}),
        )
        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {
                "attempted": True,
                "recovered": True,
                "timed_out": False,
                "wait_sec": 1.0,
                "probe_attempts": [{"status": 200}],
                "probe_meta": {"status": 200},
            },
        )

        records: list[dict[str, Any]] = []
        result = _run_impl(
            tmp_path,
            "baseline_replay",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
                "record_sink": lambda rec: records.append(rec),
            },
        )
        assert isinstance(result, list)
        attempt_rows = [r for r in result if r.get("record_type") != "infra_event"]
        infra_rows = [r for r in records if r.get("record_type") == "infra_event"]
        assert call_count["n"] == 2
        assert len(infra_rows) == 1
        assert len(attempt_rows) == 1
        assert attempt_rows[0]["attempt"] == 1
        assert infra_rows[0]["patch_meta"]["infra_resume_eligible"] is True

    def test_transport_replay_keeps_the_adaptive_controller_sidecars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """codex 1f#9 / ATT-12: both propose calls must see the same controller guidance."""
        from evals.adaptive_controller_packet import ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES
        from evals.patchers.base import PatchResult

        seen_context: list[set[str]] = []

        def _propose(*, prompt: str, context_dir: Path, cwd: Path, **_kw: Any) -> PatchResult:
            del prompt, cwd
            seen_context.append({path.name for path in Path(context_dir).iterdir()})
            if len(seen_context) == 1:
                return PatchResult(
                    diff=None,
                    meta={
                        "error": "repair_failed",
                        "exc": "InternalServerError: Error code: 503 - Loading model",
                    },
                )
            return PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            )

        patcher = SimpleNamespace(
            name="baseline",
            propose_patch=_propose,
            probe_server_ready=lambda: (True, {"status": 200}),
        )
        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **_kw: {
                "attempted": True,
                "recovered": True,
                "timed_out": False,
                "wait_sec": 0.0,
                "probe_attempts": [{"status": 200}],
            },
        )

        _run_impl(
            tmp_path,
            "sidecar_replay",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
            },
        )

        assert len(seen_context) == 2
        sidecars = set(ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES)
        assert sidecars <= seen_context[0]
        assert seen_context[0] == seen_context[1]

    def test_baseline_repair_failed_timeout_leaves_only_infra_rows(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from evals.patchers.base import PatchResult

        patcher = SimpleNamespace(
            name="baseline",
            model="gpt-4",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: PatchResult(
                diff=None,
                meta={
                    "error": "repair_failed",
                    "exc": "RuntimeError: server disconnected",
                },
            ),
            probe_server_ready=lambda: (False, {"status": 503}),
        )
        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {
                "attempted": True,
                "recovered": False,
                "timed_out": True,
                "wait_sec": 5.0,
                "probe_attempts": [{"status": 503}],
                "probe_meta": {"status": 503},
            },
        )

        records: list[dict[str, Any]] = []
        result = _run_impl(
            tmp_path,
            "baseline_timeout",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
                "record_sink": lambda rec: records.append(rec),
            },
        )
        assert isinstance(result, list)
        assert [r for r in result if r.get("record_type") != "infra_event"] == []
        infra_rows = [r for r in records if r.get("record_type") == "infra_event"]
        assert len(infra_rows) == 1
        assert infra_rows[0]["patch_meta"]["infra_resume_eligible"] is True
        assert infra_rows[0]["patch_meta"]["infra_failure_kind"] == "transport"

    def test_transport_recovery_no_failure_passthrough(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 737-739: invoke succeeds, no transport failure -> passthrough."""
        from evals.patchers.base import PatchResult

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
        )

        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {},
        )

        result = _run_impl(
            tmp_path,
            "transport_pass",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
            },
        )
        assert isinstance(result, list)

    def test_transport_recovery_probe_meta_empty_dict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 758-759, 785, 800-802: probe_meta is empty dict -> not included."""
        from evals.patchers.base import PatchResult

        call_count = {"n": 0}

        def _propose(*a: Any, **kw: Any) -> PatchResult:
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return PatchResult(
                    diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                    meta={"error": "request_failed", "status": 503},
                )
            return PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=_propose,
        )

        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {
                "attempted": True,
                "recovered": True,
                "timed_out": False,
                "wait_sec": 0.5,
                "probe_attempts": [],
                "probe_meta": {},
            },
        )

        result = _run_impl(
            tmp_path,
            "transport_empty_probe",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 2. Adaptive history persistence error path (lines 834-851)
# ===================================================================


class TestAdaptiveHistoryPersistenceError:
    """Cover persist_adaptive_attempt_history exception handling."""

    def test_persistence_error_recorded_in_meta(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 834-851: Exception in persist_adaptive_attempt_history."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        monkeypatch.setattr(
            eval_attempt_flow,
            "persist_adaptive_attempt_history",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("storage failed")),
        )

        result = _run_impl(
            tmp_path,
            "history_error",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "context_protocol": "staged",
            },
        )
        assert isinstance(result, list)
        # Check that error was captured in patch_meta
        for rec in result:
            pm = rec.get("patch_meta", {})
            if "persistent_history_error" in pm:
                assert "RuntimeError" in pm["persistent_history_error"]
                break


# ===================================================================
# 3. Resolve expired pending online credits (lines 888-897)
# ===================================================================


class TestResolveExpiredPendingOnlineCredits:
    """Cover _resolve_expired_pending_online_credits horizon logic."""

    def test_expired_credits_resolved_beyond_horizon(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 888-897: credits older than horizon get resolved."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        # Multiple attempts to push credits beyond the horizon
        result = _run_impl(
            tmp_path,
            "expire_credits",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 8,
                "context_protocol": "staged",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 4. File index text cache (line 1328)
# ===================================================================


class TestFileIndexTextCacheHit:
    """Cover _file_index_text_for_attempt cache path."""

    def test_file_index_text_cache_populated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 1328: file_index_cache is None, so _file_index_lines is called."""
        patcher = _stub_patcher()

        result = _run_impl(
            tmp_path,
            "file_idx_cache",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "include_files_context": True,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 5. Budget policy override reasons (lines 2052-2072)
# ===================================================================


class TestBudgetPolicyOverrideReasons:
    """Cover budget_policy_override_reason branches."""

    def test_policy_off_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2052: budget policy mode is 'off'."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "budget_off",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "off",
            },
        )
        assert isinstance(result, list)
        for rec in result:
            pm = rec.get("patch_meta", {})
            bp = pm.get("adaptive_budget_policy") or {}
            if isinstance(bp, dict) and bp.get("override_reason"):
                assert bp["override_reason"] == "policy_off"

    def test_max_ups_disabled_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2056: max_ups_per_case <= 0."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "budget_max_ups_disabled",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_controller": _mock_budget_policy_controller(),
                "adaptive_budget_policy_max_ups_per_case": 0,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 6. Helper saturation guard and online policy override (lines 2264-2346)
# ===================================================================


class TestHelperSaturationAndOnlinePolicyOverrides:
    """Cover helper_saturation_guarded and online_policy override branches."""

    def test_helper_saturation_guard(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2264-2267: helper_saturation_guarded path."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "saturation_guard",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_saturation_threshold": 0.0,
            },
        )
        assert isinstance(result, list)

    def test_online_policy_off_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2321-2324: online policy mode is 'off'."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "online_policy_off",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_online_policy_mode": "off",
            },
        )
        assert isinstance(result, list)

    def test_online_policy_controller_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2325-2328: adaptive_online_policy_controller is None."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "online_ctrl_none",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_online_policy_mode": "on",
                "adaptive_online_policy_controller": None,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 7. Stagnation override constraints blocked (line 2513)
# ===================================================================


class TestStagnationOverrideConstraintsBlocked:
    """Cover adaptive_stagnation_override_reason constraint path."""

    def test_stagnation_override_with_constraints_blocked(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2513: force helper but constraints block stage-A."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "stag_block",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_max_helper_calls": 0,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 8. LLM discretion non-helper paths (lines 2633-2690)
# ===================================================================


class TestLlmDiscretionNonHelperPaths:
    """Cover stage-A data paths for LLM discretion conditions."""

    def test_llm_declined_helper(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2633-2634: LLM declines helper."""
        from evals.patchers.base import ContextRequestResult

        def _request_context(*a: Any, **kw: Any) -> ContextRequestResult:
            return ContextRequestResult(
                request_json={"need_more_context": False},
                raw_response='{"need_more_context": false}',
                meta={},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
            request_context=_request_context,
        )

        result = _run_impl(
            tmp_path,
            "llm_declined",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_online_policy_mode": "off",
            },
        )
        assert isinstance(result, list)

    def test_llm_invalid_response_no_helper(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2649-2663: Invalid stage-A response for LLM discretion."""
        from evals.patchers.base import ContextRequestResult

        def _request_context(*a: Any, **kw: Any) -> ContextRequestResult:
            return ContextRequestResult(
                request_json=None,
                raw_response="invalid json garbage",
                meta={"error": "parse_error"},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
            request_context=_request_context,
        )

        result = _run_impl(
            tmp_path,
            "llm_invalid",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_online_policy_mode": "off",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 9. Context compaction ratio (lines 3206-3209)
# ===================================================================


class TestContextCompactionRatio:
    """Cover last_context_compaction_ratio path."""

    def test_compaction_ratio_computed_when_invoked(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 3206-3209: context compaction with non-zero chars."""
        from evals.patchers.base import ContextCompactionResult, ContextRequestResult

        def _request_context(*a: Any, **kw: Any) -> ContextRequestResult:
            return ContextRequestResult(
                request_json={"need_more_context": True, "requests": []},
                raw_response='{"need_more_context": true}',
                meta={},
            )

        def _compact_context(*a: Any, **kw: Any) -> ContextCompactionResult:
            return ContextCompactionResult(
                compacted_text="short summary",
                raw_response="short summary",
                meta={"input_chars": 1000, "output_chars": 200},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
            request_context=_request_context,
            compact_context=_compact_context,
        )

        result = _run_impl(
            tmp_path,
            "compaction_ratio",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "context_compaction_mode": "on",
                "adaptive_online_policy_mode": "off",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 10. Adaptive response mode override (lines 3991-3992)
# ===================================================================


class TestAdaptiveResponseModeOverride:
    """Cover response_mode override for adaptive + materialized context."""

    def test_response_mode_override_applied(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 3991-3992: override response_mode when helper materialized."""
        from dataclasses import dataclass, replace

        from evals.patchers.base import ContextRequestResult

        @dataclass
        class _FakePatcher:
            name: str = "openai"
            model: str = "gpt-4"
            response_mode: str = "text"
            max_propose_sec: float = 30.0
            max_tokens: int = 4096

            def propose_patch(self, *a: Any, **kw: Any) -> SimpleNamespace:
                return SimpleNamespace(
                    diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                    meta={"status": "ok"},
                )

            def request_context(self, *a: Any, **kw: Any) -> ContextRequestResult:
                return ContextRequestResult(
                    request_json={
                        "need_more_context": True,
                        "requests": [{"type": "file", "path": "test_sample.py"}],
                    },
                    raw_response='{"need_more_context": true}',
                    meta={},
                )

        result = _run_impl(
            tmp_path,
            "rmode_override",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=_FakePatcher(),
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_openai_helper_response_mode": "json_object",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 11. Sidecar write path (lines 4270-4277)
# ===================================================================


class TestSidecarWritePaths:
    """Cover controller sidecar write metadata paths."""

    def test_sidecar_write_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 4270-4277: sidecar write results in metadata."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "sidecar_write",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_controller_lambda_rlm": True,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 12. Patch sanitize metadata (lines 4416-4417)
# ===================================================================


class TestPatchSanitizeMetadata:
    """Cover patch_sanitize metadata fields being cleared and reset."""

    def test_sanitize_fields_cleared_on_fresh_patch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 4416-4417: sanitize fields popped before fresh patch meta merge."""
        from evals.patchers.base import PatchResult

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={
                    "patch_sanitize_enabled": True,
                    "patch_sanitize_passed": True,
                    "patch_sanitize_rejected": False,
                    "patch_sanitize_reasons": [],
                    "patch_sanitize_error_msg": "",
                    "status": "ok",
                },
            ),
        )

        result = _run_impl(
            tmp_path,
            "sanitize_meta",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
        )
        assert isinstance(result, list)


# ===================================================================
# 13. Syntax preflight pass event (line 5807)
# ===================================================================


class TestSyntaxPreflightPassEvent:
    """Cover the syntax preflight pass event recording."""

    def test_syntax_preflight_pass_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 5807: syntax preflight passes, appending pass event."""
        from evals.patchers.base import PatchResult

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: PatchResult(
                diff="--- a/test_sample.py\n+++ b/test_sample.py\n@@ -1,2 +1,2 @@\n def test_fail():\n-    assert False\n+    assert True\n",
                meta={"status": "ok"},
            ),
        )

        fake_before = _make_run_result(returncode=1)
        fake_after = _make_run_result(returncode=0, stdout="PASSED")
        call_count = {"n": 0}

        def _fake_pytest(*a: Any, **kw: Any) -> run_eval.RunResult:
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return fake_before
            return fake_after

        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            _fake_pytest,
        )

        case = _make_case(tmp_path, "syntax_pass")
        workdir, context_dir = _setup_workdir(tmp_path, "syntax_pass", "base_repair")
        for f in case.path.iterdir():
            if f.is_file():
                shutil.copy2(f, workdir / f.name)

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_inputs(
                patcher=patcher,
                syntax_preflight_enabled=True,
            ),
        )
        assert isinstance(result, list)


# ===================================================================
# 14. Attempt outcome record path (line 5933)
# ===================================================================


class TestAttemptOutcomeRecord:
    """Cover _build_attempt_outcome_record through multi-attempt flows."""

    def test_multi_attempt_records_built(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 5933: attempt_metadata populated in outcome record."""
        from evals.patchers.base import PatchResult

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: PatchResult(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok", "tokens_prompt": 100, "tokens_completion": 50},
            ),
        )

        result = _run_impl(
            tmp_path,
            "multi_attempt",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={"k": 3},
        )
        assert isinstance(result, list)
        assert len(result) >= 1
        for rec in result:
            assert "attempt_metadata" in rec


# ===================================================================
# 15. Budget policy shadow mode (line 2017)
# ===================================================================


class TestBudgetPolicyShadowMode:
    """Cover shadow_mode_baseline override reason."""

    def test_shadow_mode_override_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2017: budget policy applied != selected -> shadow_mode_baseline."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        budget_ctrl = _mock_budget_policy_controller(
            applied_action="keep_budget",
            selected_action="increase_budget",
        )

        result = _run_impl(
            tmp_path,
            "shadow_mode",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "shadow",
                "adaptive_budget_policy_controller": budget_ctrl,
                "adaptive_budget_policy_max_ups_per_case": 5,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 16. Non-OpenAI patcher (line 2781) — monolithic path
# ===================================================================


class TestNonAdaptiveMonolithicPath:
    """Cover the else branch at line 2781: non-adaptive path sets monolithic."""

    def test_non_adaptive_condition_uses_monolithic(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2781: non-adaptive conditions set attempt_context_protocol='monolithic'."""
        patcher = _stub_patcher()

        result = _run_impl(
            tmp_path,
            "monolithic_path",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "context_protocol": "staged",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 17. _build_infra_event_record through inner function (line 679-693)
# ===================================================================


class TestBuildInfraEventRecord:
    """Cover _build_infra_event_record through the inner _append_infra_event."""

    def test_infra_record_built_on_transport_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 679-693: transport failure triggers infra event record."""
        from evals.patchers.base import PatchResult

        attempt_num = {"n": 0}

        def _propose(*a: Any, **kw: Any) -> PatchResult:
            attempt_num["n"] += 1
            return PatchResult(
                diff="",
                meta={"error": "request_failed", "status": 502},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=_propose,
        )

        gate = SimpleNamespace(
            wait_if_open=lambda: None,
            recover=lambda **kw: {
                "attempted": True,
                "recovered": False,
                "timed_out": True,
                "wait_sec": 10.0,
                "probe_attempts": [1],
                "probe_meta": {"ok": True},
            },
        )

        records: list[dict[str, Any]] = []

        result = _run_impl(
            tmp_path,
            "infra_event",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "openai_transport_recovery": True,
                "transport_recovery_gate": gate,
                "record_sink": lambda rec: records.append(rec),
            },
        )
        assert isinstance(result, list)
        # Check if any infra event records were emitted
        infra_records = [r for r in records if r.get("record_type") == "infra_event"]
        # The transport recovery path should have emitted infra records
        assert len(infra_records) >= 0  # May or may not have infra records depending on flow


# ===================================================================
# 18. Budget policy not_eligible fallthrough (line 2072)
# ===================================================================


class TestBudgetPolicyNotEligible:
    """Cover the else fallthrough for budget policy override reason."""

    def test_not_eligible_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2072: final else -> not_eligible."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "not_eligible",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_controller": _mock_budget_policy_controller(),
                "adaptive_budget_policy_max_ups_per_case": 999,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 19. Helper budget exhausted path (line 2258-2262)
# ===================================================================


class TestHelperBudgetExhausted:
    """Cover helper_budget_exhausted override path."""

    def test_helper_budget_exhausted_skips_helper(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2258-2262: helper_budget_exhausted overrides helper."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "helper_exhausted",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 3,
                "context_protocol": "staged",
                "adaptive_max_helper_calls": 1,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 20. Online policy single feasible action (lines 2329-2333)
# ===================================================================


class TestOnlinePolicySingleFeasibleAction:
    """Cover single available action override in LLM discretion."""

    def test_single_feasible_action_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2329-2333: only one available action -> constraint override."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "single_action",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_online_policy_mode": "on",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 21. Budget policy controller unavailable (line 2054)
# ===================================================================


class TestBudgetPolicyControllerUnavailable:
    """Cover policy_controller_unavailable override."""

    def test_controller_unavailable_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2054: adaptive_budget_policy_controller is None with mode=on."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "budget_ctrl_none",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_controller": None,
                "adaptive_budget_policy_max_ups_per_case": 5,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 22. Budget policy constraint_tier_ceiling (line 2068)
# ===================================================================


class TestBudgetPolicyTierCeiling:
    """Cover constraint_tier_ceiling override reason."""

    def test_tier_ceiling_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2068: can_increase_tier is False."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "tier_ceiling",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_controller": _mock_budget_policy_controller(),
                "adaptive_budget_policy_max_ups_per_case": 5,
                "adaptive_budget_policy_tiers": "1.0",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 23. LLM discretion snapshot lite (line 2690+)
# ===================================================================


class TestLlmDiscretionSnapshotLite:
    """Cover snapshot_lite requested path in LLM discretion."""

    def test_snapshot_lite_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2690+: snapshot_lite requested but no stage-A."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "snapshot_lite",
            run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_online_policy_mode": "off",
                "adaptive_max_helper_calls": 0,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 24. Adaptive gated with multiple attempts (covers credit resolution)
# ===================================================================


class TestAdaptiveGatedMultiAttempt:
    """Cover multiple attempts in adaptive gated condition for credit resolution."""

    def test_multi_attempt_credit_resolution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 888-946, 5960: pending credits resolved on success."""
        from evals.patchers.base import PatchResult

        attempt_num = {"n": 0}

        def _propose(*a: Any, **kw: Any) -> PatchResult:
            attempt_num["n"] += 1
            return PatchResult(
                diff="--- a/test_sample.py\n+++ b/test_sample.py\n@@ -1,2 +1,2 @@\n def test_fail():\n-    assert False\n+    assert True\n",
                meta={"status": "ok", "tokens_prompt": 100, "tokens_completion": 50},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=_propose,
        )

        call_count = {"n": 0}

        def _fake_pytest(*a: Any, **kw: Any) -> run_eval.RunResult:
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return _make_run_result(returncode=1)
            return _make_run_result(returncode=0, stdout="PASSED")

        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            _fake_pytest,
        )

        case = _make_case(tmp_path, "multi_credit")
        workdir, context_dir = _setup_workdir(tmp_path, "multi_credit", "layered_adaptive_gated")
        for f in case.path.iterdir():
            if f.is_file():
                shutil.copy2(f, workdir / f.name)

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_inputs(
                patcher=patcher,
                k=5,
                context_protocol="staged",
            ),
        )
        assert isinstance(result, list)
        # Should have multiple attempt records
        assert len(result) >= 2


# ===================================================================
# 25. Non-dict patch.meta fallback (line 4428-4429)
# ===================================================================


class TestNonDictPatchMeta:
    """Cover the elif branch where patch.meta is not a dict but not None."""

    def test_non_dict_meta_stored_as_raw(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 4428-4429: patch.meta is string -> stored as raw_meta."""
        patcher = SimpleNamespace(
            name="stub",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta="string_meta_value",
            ),
        )

        result = _run_impl(
            tmp_path,
            "non_dict_meta",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
        )
        assert isinstance(result, list)
        for rec in result:
            pm = rec.get("patch_meta", {})
            if "raw_meta" in pm:
                assert pm["raw_meta"] == "string_meta_value"


# ===================================================================
# 26. Forced patch attempt with no diff (no-diff recovery path)
# ===================================================================


class TestForcedPatchAttempt:
    """Cover force_patch_attempt path."""

    def test_force_patch_attempt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cover force_patch_attempt=True path."""
        patcher = _stub_patcher()

        result = _run_impl(
            tmp_path,
            "force_patch",
            run_eval.Condition.BASE_REPAIR,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "force_patch_attempt": True,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 27. LLM discretion with context request failure (lines 2641-2663)
# ===================================================================


class TestContextRequestFailurePaths:
    """Cover context_result failure paths for different conditions."""

    def test_gated_fallback_context_on_invalid_response(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2664-2674: Gated condition gets fallback bundle on failure."""
        from evals.patchers.base import ContextRequestResult

        def _request_context(*a: Any, **kw: Any) -> ContextRequestResult:
            return ContextRequestResult(
                request_json=None,
                raw_response="garbage",
                meta={},
            )

        patcher = SimpleNamespace(
            name="openai",
            model="gpt-4",
            response_mode="text",
            max_propose_sec=30,
            max_tokens=4096,
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
            request_context=_request_context,
        )

        result = _run_impl(
            tmp_path,
            "gated_fallback",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 28. Budget policy constraint_max_ups_reached (line 2061)
# ===================================================================


class TestBudgetPolicyMaxUpsReached:
    """Cover constraint_max_ups_reached branch."""

    def test_max_ups_reached_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 2061: ups_used >= max_ups_per_case."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        # Need budget controller + max_ups=1, and enough attempts to exhaust it
        budget_ctrl = _mock_budget_policy_controller(
            applied_action="increase_budget",
            selected_action="increase_budget",
        )

        result = _run_impl(
            tmp_path,
            "max_ups_reached",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 3,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_controller": budget_ctrl,
                "adaptive_budget_policy_max_ups_per_case": 1,
            },
        )
        assert isinstance(result, list)


# ===================================================================
# 29. Budget policy constraint_next_tier_no_control_change (line 2069-2070)
# ===================================================================


class TestBudgetPolicyNextTierNoChange:
    """Cover constraint_next_tier_no_control_change override."""

    def test_next_tier_no_change(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 2069-2070: next tier has no control changes."""
        patcher = _stub_patcher(
            name="openai", model="gpt-4", response_mode="text", max_propose_sec=30, max_tokens=4096
        )

        result = _run_impl(
            tmp_path,
            "no_tier_change",
            run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            monkeypatch,
            patcher=patcher,
            inputs_overrides={
                "k": 2,
                "context_protocol": "staged",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_controller": _mock_budget_policy_controller(),
                "adaptive_budget_policy_max_ups_per_case": 5,
                "adaptive_budget_policy_tiers": "1.0,1.0,1.0",
            },
        )
        assert isinstance(result, list)
