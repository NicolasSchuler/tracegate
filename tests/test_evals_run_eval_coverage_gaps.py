"""Coverage-gap tests for evals.run_eval — compact, targeted at uncovered lines."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import run_eval
from evals.eval_types import (
    EVAL_RECORD_TYPE_ATTEMPT,
    EVAL_RECORD_TYPE_INFRA_EVENT,
    CaseSpec,
    Condition,
    ConfigResolutionState,
    RunResult,
)
from evals.patchers.base import (
    ContextCompactionResult,
    ContextRequestResult,
    PatchPlanResult,
    PatchResult,
)

_ns = argparse.Namespace


def _run(
    *, rc: int = 1, out: str = "", err: str = "", to: bool = False, ts: int | None = None
) -> RunResult:
    return RunResult(
        returncode=rc, duration_sec=0.1, stdout=out, stderr=err, timed_out=to, timeout_sec=ts
    )


def _case(cid: str = "c") -> CaseSpec:
    return CaseSpec(
        case_id=cid,
        path=Path("/f"),
        category="runtime",
        difficulty="easy",
        snapshot_dependency="full",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "t.py"],
        timeout_sec=60,
    )


# === _OpenAITransportRecoveryGate ===========================================


class TestRecoveryGate:
    def test_init_and_wait_if_open(self) -> None:
        g = run_eval._OpenAITransportRecoveryGate(enabled=True, max_wait_sec=5, poll_sec=0.05)
        assert g._enabled and g._max_wait_sec == 5.0 and not g._outage_active
        g._outage_active = True
        released = threading.Event()
        threading.Thread(target=lambda: (g.wait_if_open(), released.set()), daemon=True).start()
        time.sleep(0.08)
        assert not released.is_set()
        with g._condition:
            g._outage_active = False
            g._condition.notify_all()
        released.wait(2)
        assert released.is_set()

    def test_leader_probe_retry_then_success(self) -> None:
        n = 0

        class P:
            name = "openai"

            def probe_server_ready(self):
                nonlocal n
                n += 1
                return (n >= 2), {"server_ready_probe_provider": "s"}

        o = run_eval._OpenAITransportRecoveryGate(
            enabled=True, max_wait_sec=5, poll_sec=0.02
        ).recover(patcher=P(), operation_name="op")
        assert o["recovered"] and o["leader"] and len(o["probe_attempts"]) >= 2

    def test_leader_probe_exception(self) -> None:
        class P:
            name = "openai"

            def probe_server_ready(self):
                raise ConnectionError("boom")

        o = run_eval._OpenAITransportRecoveryGate(
            enabled=True, max_wait_sec=0.15, poll_sec=0.03
        ).recover(patcher=P(), operation_name="op")
        assert o["timed_out"] and o["leader"] and len(o["probe_attempts"]) >= 1

    def test_follower_waits(self) -> None:
        g = run_eval._OpenAITransportRecoveryGate(enabled=True, max_wait_sec=5, poll_sec=0.02)
        g._outage_active, g._outage_generation, g._leader_claimed_generation = True, 1, 1
        g._outcomes[1] = {
            "attempted": True,
            "recovered": True,
            "timed_out": False,
            "wait_sec": 0.1,
        }
        threading.Thread(
            target=lambda: (
                time.sleep(0.03),
                setattr(g, "_outage_active", False),
                g._condition.acquire(),
                g._condition.notify_all(),
                g._condition.release(),
            ),
            daemon=True,
        ).start()
        o = g.recover(patcher=SimpleNamespace(name="openai"), operation_name="t")
        assert not o["leader"] and o["attempted"]


# === _build_infra_event_record ===============================================


class TestBuildInfraEvent:
    def test_both_metadata_paths(self) -> None:
        c, b, p = (
            _case(),
            _run(rc=1, to=True, ts=60),
            SimpleNamespace(name="openai", model="m", base_url="u"),
        )
        for am in (None, {"attempt": 2, "x": 1}):
            r = run_eval._build_infra_event_record(
                case=c,
                condition=Condition.BASE_REPAIR,
                run_id="r",
                attempt=1,
                patcher_name="openai",
                run_metadata={"run_id": "r"},
                attempt_metadata=am,
                patcher=p,
                patch_meta={"error": "t"},
                context_protocol="monolithic",
                before=b,
                cumulative_runtime_sec=1.5,
            )
            assert r["record_type"] == EVAL_RECORD_TYPE_INFRA_EVENT and r["success"] is False


# === _record_token_usage_state ===============================================


class TestTokenUsageState:
    @pytest.mark.parametrize(
        "am,pm,pt,tt,expected",
        [
            (None, None, 10, 20, "available"),
            ({"usage_available": True, "llm_call_count": 1}, None, 0, 0, "available"),
            ({"usage_available": False, "llm_call_count": 1}, None, 0, 0, "unavailable"),
            ({"usage_available": False, "llm_call_count": 0}, None, 0, 0, "not_applicable"),
            ({"llm_call_count": 2}, None, 0, 0, "unavailable"),
            ({"error": "timeout"}, None, 0, 0, "unavailable"),
            (None, None, 0, 0, "not_applicable"),
        ],
    )
    def test_all_branches(self, am: Any, pm: Any, pt: int, tt: int, expected: str) -> None:
        assert (
            run_eval._record_token_usage_state(
                attempt_meta=am, patch_meta=pm, prompt_tokens=pt, total_tokens=tt
            )
            == expected
        )


# === _format_progress_token_usage ============================================


class TestFormatProgressTokenUsage:
    @pytest.mark.parametrize(
        "avail,unavail,expect_in",
        [
            (0, 1, "tokens=unavailable"),
            (0, 0, "tokens=n/a"),
            (1, 1, "coverage=partial"),
            (2, 0, "prompt:10"),
        ],
    )
    def test_branches(self, avail: int, unavail: int, expect_in: str) -> None:
        r = run_eval._format_progress_token_usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            available_records=avail,
            unavailable_records=unavail,
        )
        assert expect_in in r


# === _aggregate_attempt_llm_usage ============================================


def test_aggregate_llm_usage() -> None:
    sa = {"llm_call_count": 1, "tokens_prompt": 100, "tokens_completion": 50, "tokens_total": 150}
    pm = {"llm_call_count": 2, "tokens_prompt": 200, "tokens_completion": 80, "tokens_total": 280}
    c, p, co, t = run_eval._aggregate_attempt_llm_usage(
        stage_a_meta=sa, patch_meta=pm, stage_a_default_llm_calls=0, patch_default_llm_calls=0
    )
    assert (c, p, co, t) == (3, 300, 130, 430)
    c2, *_ = run_eval._aggregate_attempt_llm_usage(
        stage_a_meta=None, patch_meta=None, stage_a_default_llm_calls=1, patch_default_llm_calls=1
    )
    assert c2 == 2


# === _build_attempt_budget_exhausted_plan_result =============================


def test_budget_exhausted_plan() -> None:
    p = SimpleNamespace(name="openai", model="m", base_url="u", response_mode="e")
    r = run_eval._build_attempt_budget_exhausted_plan_result(patcher=p, remaining_budget_sec=-1.0)
    assert (
        isinstance(r, PatchPlanResult)
        and r.meta["error"] == "request_budget_exceeded"
        and r.meta["patcher"] == "openai"
    )
    r2 = run_eval._build_attempt_budget_exhausted_plan_result(
        patcher=SimpleNamespace(), remaining_budget_sec=0
    )
    assert "patcher" not in r2.meta


# === _status_code_from_wait_result ===========================================


@pytest.mark.parametrize(
    "inp,exp",
    [
        (0, 0),
        (42, 42),
        ({"StatusCode": 0}, 0),
        ({"StatusCode": "2"}, 2),
        ({"StatusCode": "nope"}, 1),
        ("x", 1),
        (None, 1),
    ],
)
def test_status_code_from_wait(inp: Any, exp: int) -> None:
    assert run_eval._status_code_from_wait_result(inp) == exp


# === _is_container_wait_timeout ==============================================


def test_container_wait_timeout() -> None:
    assert run_eval._is_container_wait_timeout(type("ReadTimeoutError", (Exception,), {})("x"))
    assert run_eval._is_container_wait_timeout(Exception("read timed out"))
    assert not run_eval._is_container_wait_timeout(Exception("connection error"))


# === _request_context_failure_kind_from_meta =================================


class TestReqCtxFailureKind:
    @pytest.mark.parametrize(
        "meta,expected",
        [
            ({"request_context_failure_kind": "custom"}, "custom"),
            ({"error": "request_budget_exceeded"}, "request_budget_exceeded"),
            ({"request_context_backend_unavailable": True}, "backend_unavailable"),
            ({"error": "invalid_request_context_payload"}, "invalid_payload"),
            ({"error": "request_failed"}, "transport"),
            ({"error": "http_error", "status": 502}, "backend_unavailable"),
            ({"error": "http_error", "status": 400}, "transport"),
            ({}, ""),
            (None, ""),
        ],
    )
    def test_all(self, meta: Any, expected: str) -> None:
        assert run_eval._request_context_failure_kind_from_meta(meta) == expected


# === _request_context_recovery_failure_kind_from_meta ========================


class TestReqCtxRecoveryKind:
    @pytest.mark.parametrize(
        "meta,expected",
        [
            ({"error": "request_failed"}, "transport"),
            ({"error": "http_error", "status": 503}, "backend_unavailable"),
            ({"error": "http_error", "status": 429}, "transport"),
            ({"error": "http_error", "status": 400}, ""),
        ],
    )
    def test_all(self, meta: dict, expected: str) -> None:
        assert run_eval._request_context_recovery_failure_kind_from_meta(meta) == expected


# === _record_is_transport_request_failure ====================================


class TestRecordTransportFailure:
    @pytest.mark.parametrize(
        "record,expected",
        [
            (
                {
                    "record_type": EVAL_RECORD_TYPE_ATTEMPT,
                    "note": "patcher_request_failed",
                    "patch_meta": {
                        "patcher_request_failed": True,
                        "patcher_request_failure_kind": "transport",
                    },
                },
                True,
            ),
            (
                {
                    "record_type": EVAL_RECORD_TYPE_ATTEMPT,
                    "note": "patcher_request_failed",
                    "patch_meta": {
                        "patcher_request_failed": True,
                        "patcher_request_failure_kind": "context_limit",
                    },
                },
                False,
            ),
            (
                {"record_type": EVAL_RECORD_TYPE_INFRA_EVENT, "note": "patcher_request_failed"},
                False,
            ),
            ({"record_type": EVAL_RECORD_TYPE_ATTEMPT, "note": "success", "patch_meta": {}}, False),
            (
                {
                    "record_type": EVAL_RECORD_TYPE_ATTEMPT,
                    "note": "patcher_request_failed",
                    "patch_meta": "x",
                },
                False,
            ),
            (
                {
                    "record_type": EVAL_RECORD_TYPE_ATTEMPT,
                    "note": "patcher_request_failed",
                    "patch_meta": {"error": "request_failed"},
                },
                True,
            ),
        ],
    )
    def test_all(self, record: dict, expected: bool) -> None:
        assert run_eval._record_is_transport_request_failure(record) is expected


# === _adaptive_pipeline_stage_before =========================================


class TestPipelineStageBefore:
    @pytest.mark.parametrize(
        "prev,cur,stage",
        [
            ({"note": "patch_apply_failed"}, None, "pretest_patch_failure"),
            (None, _run(rc=1, to=True, ts=60), "tests_still_failing"),
            (None, _run(rc=0), "success"),
            (None, None, "tests_still_failing"),
        ],
    )
    def test_all(self, prev: Any, cur: Any, stage: str) -> None:
        assert (
            run_eval._adaptive_pipeline_stage_before(
                previous_failure_context=prev, current_run=cur
            )[0]
            == stage
        )


# === _normalize_adaptive_controller_lambda_rlm_settings ======================


class TestNormAdaptiveCtrlLambda:
    def test_disabled(self) -> None:
        a = _ns(adaptive_controller_lambda_rlm=False)
        run_eval._normalize_adaptive_controller_lambda_rlm_settings(args=a, config_state=None)
        assert not hasattr(a, "adaptive_controller_lambda_rlm_repo_dir")

    def test_fills_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "evals.lambda_rlm_backend.default_lambda_rlm_repo_dir", lambda: Path("/r")
        )
        monkeypatch.setattr("evals.baseline_runtime._autodetect_lambda_rlm_python", lambda: "/p")
        monkeypatch.setattr(
            "evals.baseline_runtime._baseline_runtime_manifest_path", lambda *_a, **_k: None
        )
        a = _ns(
            adaptive_controller_lambda_rlm=True,
            adaptive_controller_lambda_rlm_repo_dir="",
            adaptive_controller_lambda_rlm_python="",
        )
        cs = ConfigResolutionState(config_paths=[], value_sources={})
        run_eval._normalize_adaptive_controller_lambda_rlm_settings(args=a, config_state=cs)
        assert a.adaptive_controller_lambda_rlm_repo_dir == "/r"
        assert a.adaptive_controller_lambda_rlm_python == "/p"


# === _build_pytest_args ======================================================


class TestBuildPytestArgs:
    def test_snapshot_injects_plugin(self) -> None:
        a = run_eval._build_pytest_args(
            python_cmd="python",
            python_args=["-m", "pytest", "t.py"],
            condition=Condition.LAYERED_ADAPTIVE_GATED,
        )
        assert "llmdebug.pytest_plugin" in a
        a2 = run_eval._build_pytest_args(
            python_cmd="python", python_args=["t.py"], condition=Condition.LAYERED_ADAPTIVE_LLM
        )
        assert "llmdebug.pytest_plugin" in a2

    def test_non_snapshot(self) -> None:
        a = run_eval._build_pytest_args(
            python_cmd="python", python_args=["-m", "pytest"], condition=Condition.BASE_REPAIR
        )
        assert "-p" not in a


# === _summarize_context_compaction_meta ======================================


def test_summarize_compaction_meta() -> None:
    out = run_eval._summarize_context_compaction_meta(
        {
            "llm_call_count": 1,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "status": 200,
            "error": "err",
            "body": "b",
        }
    )
    assert out["usage"]["prompt_tokens"] == 10 and out["status"] == 200 and out["error"] == "err"


# === _adaptive_prompt_budget_for_attempt =====================================


class TestAdaptiveBudgetCompact:
    def test_compact_caps(self) -> None:
        b = {
            "total_chars": 500_000,
            "retry_delta_chars": 50_000,
            "stdout_chars": 50_000,
            "snapshot_chars": 50_000,
            "evidence_chars": 100_000,
            "retry_history_chars": 50_000,
        }
        r = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget=b, helper_materialized_context=True, profile="compact"
        )
        assert r["total_chars"] == 120_000 and r["evidence_chars"] == 32_000

    def test_non_compact_passthrough(self) -> None:
        r = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget={"total_chars": 500_000},
            helper_materialized_context=True,
            profile="default",
        )
        assert r["total_chars"] == 500_000

    def test_zero_not_capped(self) -> None:
        r = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget={"total_chars": 0, "retry_delta_chars": -1},
            helper_materialized_context=True,
            profile="compact",
        )
        assert r["total_chars"] == 0 and r["retry_delta_chars"] == -1


# === _plan_patch_with_snapshot_context =======================================


class TestPlanPatchSnapshot:
    def test_budget_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "evals.attempt_proposal._patcher_with_shared_attempt_budget",
            lambda *, patcher, attempt_deadline_monotonic: (patcher, -1.0, None, False),
        )
        r = run_eval._plan_patch_with_snapshot_context(
            patcher=SimpleNamespace(name="openai", model="m", base_url="u", response_mode="e"),
            prompt="f",
            context_dir=Path("/f"),
            cwd=Path("/f"),
            snapshot_full_text=None,
            snapshot_artifact_text=None,
            attempt_deadline_monotonic=0,
        )
        assert isinstance(r, PatchPlanResult) and r.meta["error"] == "request_budget_exceeded"

    def test_swap_and_enrich(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "evals.attempt_proposal._patcher_with_shared_attempt_budget",
            lambda *, patcher, attempt_deadline_monotonic: (patcher, 10, 5, True),
        )
        sf = tmp_path / "snapshot.json"
        sf.write_text("art", encoding="utf-8")

        class P:
            name = "openai"

            def plan_patch(self, *, prompt, context_dir, cwd):
                return PatchPlanResult(
                    plan_text="p",
                    raw_response="r",
                    meta={"read": (context_dir / "snapshot.json").read_text()},
                )

        r = run_eval._plan_patch_with_snapshot_context(
            patcher=P(),
            prompt="f",
            context_dir=tmp_path,
            cwd=tmp_path,
            snapshot_full_text="full",
            snapshot_artifact_text="art",
        )
        assert (
            r.meta["read"] == "full"
            and sf.read_text() == "art"
            and r.meta.get("attempt_shared_propose_budget_clamped")
        )


# === _propose_patch_with_snapshot_context ====================================


class TestProposePatchSnapshot:
    def test_swap_and_restore(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "evals.attempt_proposal._patcher_with_shared_attempt_budget",
            lambda *, patcher, attempt_deadline_monotonic: (patcher, 10, None, False),
        )
        sf = tmp_path / "snapshot.json"
        sf.write_text("art", encoding="utf-8")

        class P:
            name = "openai"

            def propose_patch(self, *, prompt, context_dir, cwd):
                return PatchResult(
                    diff="d", meta={"read": (context_dir / "snapshot.json").read_text()}
                )

        r = run_eval._propose_patch_with_snapshot_context(
            patcher=P(),
            prompt="f",
            context_dir=tmp_path,
            cwd=tmp_path,
            snapshot_full_text="full",
            snapshot_artifact_text="art",
        )
        assert r.meta["read"] == "full" and sf.read_text() == "art"

    def test_no_swap_equal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "evals.attempt_proposal._patcher_with_shared_attempt_budget",
            lambda *, patcher, attempt_deadline_monotonic: (patcher, None, None, False),
        )

        class P:
            name = "openai"

            def propose_patch(self, *, prompt, context_dir, cwd):
                return PatchResult(diff="d", meta={"ok": True})

        r = run_eval._propose_patch_with_snapshot_context(
            patcher=P(),
            prompt="f",
            context_dir=tmp_path,
            cwd=tmp_path,
            snapshot_full_text="s",
            snapshot_artifact_text="s",
        )
        assert r.diff == "d"


# === load_cases ==============================================================


class TestLoadCases:
    def test_missing_dir(self, tmp_path: Path) -> None:
        assert run_eval.load_cases(tmp_path / "nope") == []

    def test_invalid_json(self, tmp_path: Path) -> None:
        d = tmp_path / "a"
        d.mkdir()
        (d / "case.json").write_text("{bad", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            run_eval.load_cases(tmp_path)

    def test_missing_category(self, tmp_path: Path) -> None:
        d = tmp_path / "b"
        d.mkdir()
        (d / "case.json").write_text(
            '{"difficulty":"easy","snapshot_dependency":"full","expected_exception":"E"}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="category"):
            run_eval.load_cases(tmp_path)

    def test_files_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("x", encoding="utf-8")
        assert run_eval.load_cases(tmp_path) == []


# === _build_structured_previous_attempt_diff =================================


class TestStructuredDiff:
    def test_none_for_no_prev(self) -> None:
        assert (
            run_eval._build_structured_previous_attempt_diff(
                None, current_failure_context={"attempt": 1}
            )
            is None
        )

    def test_snapshot_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("llmdebug.snapshot_diff.diff_snapshots", lambda a, b: {"changed": True})
        monkeypatch.setattr("llmdebug.snapshot_diff.structured_diff_summary", lambda d: {"s": 1})
        r = run_eval._build_structured_previous_attempt_diff(
            {"attempt": 1, "failure_signature": "s1", "snapshot": {"a": 1}},
            current_failure_context={"attempt": 2, "failure_signature": "s2", "snapshot": {"b": 2}},
        )
        assert r and r["attempt_prev"] == "attempt_01"


# === _score_patch_relevance ==================================================


class TestScorePatchRelevance:
    def test_no_crash(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("evals.patch_relevance._import_neighbors", lambda **kw: [])
        monkeypatch.setattr("evals.patch_relevance._crash_file_from_snapshot", lambda s: "")
        monkeypatch.setattr("evals.patch_relevance._detect_failing_test_file", lambda **kw: "")
        monkeypatch.setattr("evals.patch_relevance._infer_test_source_scope", lambda **kw: [])
        _, d = run_eval._score_patch_relevance(
            diff="diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n",
            cwd=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
        )
        assert "patch_verifier_reasons" in d

    def test_touches_test(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("evals.patch_relevance._import_neighbors", lambda **kw: [])
        monkeypatch.setattr("evals.patch_relevance._crash_file_from_snapshot", lambda s: "bar.py")
        monkeypatch.setattr(
            "evals.patch_relevance._detect_failing_test_file", lambda **kw: "test_f.py"
        )
        monkeypatch.setattr("evals.patch_relevance._infer_test_source_scope", lambda **kw: [])
        _, d = run_eval._score_patch_relevance(
            diff="diff --git a/test_f.py b/test_f.py\n--- a/test_f.py\n+++ b/test_f.py\n@@ -1 +1 @@\n-a\n+b\n",
            cwd=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
        )
        assert "touches_failing_test_file" in d["patch_verifier_reasons"]


# === make_patcher ============================================================


class TestMakePatcher:
    def _args(self, **kw: Any) -> argparse.Namespace:
        d = {
            "patcher": "null",
            "baseline": "",
            "baseline_name": "",
            "baseline_repo_dir": "",
            "baseline_python": "",
            "adaptive_controller_lambda_rlm": False,
            "patch_cmd": "",
            "patch_cmd_shell": False,
            "patch_cmd_timeout_sec": 30,
            "openai_model": "",
            "openai_api_key": "",
            "openai_base_url": "http://x/v1",
            "openai_timeout_sec": 300,
            "openai_http2": True,
            "openai_transport_max_retries": 3,
            "openai_transport_retry_backoff_base_sec": 1.0,
            "openai_transport_retry_backoff_max_sec": 30.0,
            "openai_max_propose_sec": 120,
            "openai_max_tokens": 4096,
            "openai_temperature": 0.0,
            "openai_compat_profile": "lmstudio_model_yaml",
            "openai_allow_reasoning_fallback": True,
            "openai_enable_thinking": True,
            "openai_response_mode": "edits",
            "openai_force_patch_attempt": False,
            "deterministic": False,
            "seed": None,
        }
        d.update(kw)
        return _ns(**d)

    def test_null(self) -> None:
        from evals.patchers import NullPatcher

        assert isinstance(run_eval.make_patcher(self._args(patcher="null")), NullPatcher)

    def test_heuristic(self) -> None:
        from evals.patchers import HeuristicSnapshotPatcher

        assert isinstance(
            run_eval.make_patcher(self._args(patcher="heuristic")), HeuristicSnapshotPatcher
        )

    def test_openai_no_model(self) -> None:
        with pytest.raises(SystemExit, match="openai-model"):
            run_eval.make_patcher(self._args(patcher="openai"))

    def test_openai_with_model(self) -> None:
        from evals.patchers import OpenAICompatPatcher

        assert isinstance(
            run_eval.make_patcher(self._args(patcher="openai", openai_model="m")),
            OpenAICompatPatcher,
        )

    def test_openai_with_model_uses_normalized_compat_settings(self) -> None:
        patcher = run_eval.make_patcher(self._args(patcher="openai", openai_model="m"))

        assert patcher.compat_profile == "lmstudio_model_yaml"
        assert patcher.allow_reasoning_fallback is True
        assert patcher.enable_thinking is True

    def test_openai_thinking_auto_enables_reasoning_fallback(self) -> None:
        patcher = run_eval.make_patcher(
            self._args(
                patcher="openai",
                openai_model="m",
                openai_allow_reasoning_fallback=False,
                openai_enable_thinking=True,
            )
        )

        assert patcher.allow_reasoning_fallback is True
        assert patcher.enable_thinking is True

    def test_command(self) -> None:
        from evals.patchers import CommandPatcher

        assert isinstance(
            run_eval.make_patcher(self._args(patcher="command", patch_cmd="echo")), CommandPatcher
        )

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evals.patchers import NullPatcher

        monkeypatch.setattr(
            "evals.patchers.command_patcher.patcher_from_env", lambda: NullPatcher()
        )
        assert isinstance(run_eval.make_patcher(self._args(patcher="unk")), NullPatcher)

    def test_baseline_no_name(self) -> None:
        with pytest.raises(SystemExit, match="baseline-name"):
            run_eval.make_patcher(self._args(patcher="baseline", baseline=""))
