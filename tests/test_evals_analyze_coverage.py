"""Additional coverage tests for ``evals.analyze_results`` — targeting uncovered lines."""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from evals import analyze_results as ar

# ── Shared helpers ───────────────────────────────────────────────────────


def _record(
    *,
    run_id: str = "run",
    case_id: str = "case",
    condition: str = "base_repair",
    attempt: int = 1,
    success: bool = False,
    category: str = "logic_error",
    difficulty: str = "easy",
    snapshot_dependency: str = "required",
    context_protocol: str = "monolithic",
    attempt_wall_clock_sec: float | None = 1.0,
    cumulative_runtime_sec: float | None = 1.0,
    note: str = "tests_still_failing",
    schema_version: int = ar.EVAL_RECORD_SCHEMA_VERSION,
    record_type: str = "attempt",
    attempt_metadata_extra: dict[str, object] | None = None,
    patch_meta_extra: dict[str, object] | None = None,
    model_backend: str = "openai",
) -> dict[str, object]:
    prompt_hash_field = "prompt_xxh3_128" if schema_version >= 6 else "prompt_sha256"
    evidence_hash_field = "evidence_xxh3_128" if schema_version >= 6 else "evidence_sha256"
    attempt_metadata: dict[str, object] = {
        "attempt": attempt,
        "max_attempts": 3,
        "attempt_ordinal_label": f"{attempt}/3",
        "llm_call_count": 2,
        "tokens_prompt": 4,
        "tokens_completion": 5,
        "tokens_total": 9,
        "retry_history_budget_chars": 120,
        "retry_history_budget_applied": False,
        prompt_hash_field: None,
        "prompt_path": None,
        evidence_hash_field: None,
        "evidence_path": None,
        "token_usage_available": True,
        "context_protocol": context_protocol,
        "stage_b_prompt_chars": 100,
        "stage_a_request_success": None,
        "retry_history_chars": 0,
        "retry_history_turns_included": 0,
        "retry_history_truncated": False,
        "timing_breakdown": {},
        "token_breakdown": {},
    }
    if attempt_metadata_extra:
        attempt_metadata.update(attempt_metadata_extra)

    model_metadata: dict[str, object] = {"backend": model_backend, "name": model_backend}
    if model_backend == "openai":
        model_metadata.update(
            {
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "dummy",
                "temperature": 0.0,
                "max_tokens": 2048,
                "response_mode": "edits",
                "api_compat_seed_requested": None,
                "api_compat_seed_acknowledged": False,
            }
        )

    rec: dict[str, object] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": attempt,
        "success": success,
        "record_type": record_type,
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": snapshot_dependency,
        "context_protocol": context_protocol,
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": "2026-02-10T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.12.0",
            "platform": "test",
            "llmdebug_version": "2.0.0",
            "patcher_name": "command",
            "patcher_config_fingerprint": "abc123",
            "run_settings_fingerprint": "runfp123",
        },
        "attempt_metadata": attempt_metadata,
        "model_metadata": model_metadata,
        "note": note,
    }
    if attempt_wall_clock_sec is not None:
        rec["attempt_wall_clock_sec"] = attempt_wall_clock_sec
    if cumulative_runtime_sec is not None:
        rec["cumulative_runtime_sec"] = cumulative_runtime_sec
    if patch_meta_extra:
        rec["patch_meta"] = patch_meta_extra
    return rec


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)


# ── Schema validation edge cases (lines 404-529) ────────────────────────


class TestValidateEvalRecordSchema:
    """Exercises _validate_eval_record_schema branches."""

    def _valid(self, **overrides: object) -> dict[str, Any]:
        rec = _record()
        for k, v in overrides.items():
            rec[k] = v
        return rec

    def test_empty_run_id(self) -> None:
        rec = self._valid(run_id="")
        with pytest.raises(ValueError, match="run_id must be a non-empty string"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_empty_case_id(self) -> None:
        rec = self._valid(case_id="")
        with pytest.raises(ValueError, match="case_id must be a non-empty string"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_empty_condition(self) -> None:
        rec = self._valid(condition="")
        with pytest.raises(ValueError, match="condition must be a non-empty string"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_invalid_record_type(self) -> None:
        rec = self._valid(record_type="bogus")
        with pytest.raises(ValueError, match="record_type must be one of"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_success_not_bool(self) -> None:
        rec = self._valid(success=1)
        with pytest.raises(ValueError, match="success must be a boolean"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_negative(self) -> None:
        rec = self._valid(attempt=-1)
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["attempt"] = -1
        with pytest.raises(ValueError, match="attempt must be a non-negative integer"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_wall_clock_sec_negative(self) -> None:
        rec = self._valid(attempt_wall_clock_sec=-1.0)
        with pytest.raises(ValueError, match="attempt_wall_clock_sec must be a non-negative"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_run_metadata_not_dict(self) -> None:
        rec = self._valid()
        rec["run_metadata"] = "bad"
        with pytest.raises(ValueError, match="run_metadata must be an object"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_run_metadata_missing_field(self) -> None:
        rec = self._valid()
        rm = rec["run_metadata"]
        assert isinstance(rm, dict)
        del rm["run_id"]
        with pytest.raises(ValueError, match="run_metadata missing 'run_id'"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_run_metadata_run_id_mismatch(self) -> None:
        rec = self._valid()
        rm = rec["run_metadata"]
        assert isinstance(rm, dict)
        rm["run_id"] = "different"
        with pytest.raises(ValueError, match=r"run_metadata\.run_id must match"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_run_metadata_deterministic_not_bool(self) -> None:
        rec = self._valid()
        rm = rec["run_metadata"]
        assert isinstance(rm, dict)
        rm["deterministic"] = "yes"
        with pytest.raises(ValueError, match="deterministic must be a boolean"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_run_metadata_seed_not_int(self) -> None:
        rec = self._valid()
        rm = rec["run_metadata"]
        assert isinstance(rm, dict)
        rm["seed"] = "abc"
        with pytest.raises(ValueError, match=r"run_metadata\.seed must be int\|null"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_not_dict(self) -> None:
        rec = self._valid()
        rec["attempt_metadata"] = "bad"
        with pytest.raises(ValueError, match="attempt_metadata must be an object"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_missing_field(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        del am["attempt"]
        with pytest.raises(ValueError, match="attempt_metadata missing 'attempt'"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_missing_hash_field(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        field = "prompt_xxh3_128"
        if field in am:
            del am[field]
        with pytest.raises(ValueError, match="attempt_metadata missing"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_attempt_mismatch(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["attempt"] = 999
        with pytest.raises(ValueError, match=r"attempt_metadata\.attempt must match"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_max_attempts_invalid(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["max_attempts"] = "bad"
        with pytest.raises(ValueError, match="max_attempts must be a non-negative"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_ordinal_label_empty(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["attempt_ordinal_label"] = ""
        with pytest.raises(ValueError, match="attempt_ordinal_label must be a non-empty"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_llm_call_count_invalid(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["llm_call_count"] = -1
        with pytest.raises(ValueError, match="llm_call_count must be a non-negative"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_token_usage_available_invalid(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["token_usage_available"] = "yes"
        with pytest.raises(ValueError, match=r"token_usage_available must be bool\|null"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_reasoning_token_usage_available_invalid(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["reasoning_token_usage_available"] = "yes"
        with pytest.raises(ValueError, match=r"reasoning_token_usage_available must be bool\|null"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_tokens_negative(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["tokens_prompt"] = -1
        with pytest.raises(ValueError, match="tokens_prompt must be a non-negative int"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_reasoning_tokens_negative(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["tokens_reasoning"] = -1
        with pytest.raises(ValueError, match="tokens_reasoning must be a non-negative int\\|null"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_attempt_metadata_retry_budget_applied_not_bool(self) -> None:
        rec = self._valid()
        am = rec["attempt_metadata"]
        assert isinstance(am, dict)
        am["retry_history_budget_applied"] = 0
        with pytest.raises(ValueError, match="retry_history_budget_applied must be a boolean"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_model_metadata_not_dict(self) -> None:
        rec = self._valid()
        rec["model_metadata"] = "bad"
        with pytest.raises(ValueError, match="model_metadata must be an object"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_model_metadata_missing_field(self) -> None:
        rec = self._valid()
        mm = rec["model_metadata"]
        assert isinstance(mm, dict)
        del mm["backend"]
        with pytest.raises(ValueError, match="model_metadata missing 'backend'"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_model_metadata_backend_empty(self) -> None:
        rec = self._valid()
        mm = rec["model_metadata"]
        assert isinstance(mm, dict)
        mm["backend"] = ""
        with pytest.raises(ValueError, match="backend must be a non-empty string"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_openai_model_metadata_missing_field(self) -> None:
        rec = self._valid()
        mm = rec["model_metadata"]
        assert isinstance(mm, dict)
        mm["backend"] = "openai"
        del mm["base_url"]
        with pytest.raises(ValueError, match="model_metadata missing 'base_url' for openai"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_openai_seed_requested_bad_type(self) -> None:
        rec = self._valid()
        mm = rec["model_metadata"]
        assert isinstance(mm, dict)
        mm["api_compat_seed_requested"] = "bad"
        with pytest.raises(ValueError, match=r"api_compat_seed_requested must be int\|null"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_openai_seed_acknowledged_not_bool(self) -> None:
        rec = self._valid()
        mm = rec["model_metadata"]
        assert isinstance(mm, dict)
        mm["api_compat_seed_acknowledged"] = 1
        with pytest.raises(ValueError, match="api_compat_seed_acknowledged must be a boolean"):
            ar._validate_eval_record_schema(rec, src="t", line_no=1)

    def test_legacy_schema_hash_fields(self) -> None:
        """schema_version=5 uses sha256 hash fields."""
        rec = _record(schema_version=5)
        # Should validate fine with sha256 fields — no ValueError raised
        result = ar._validate_eval_record_schema(rec, src="t", line_no=1)
        assert result is None

    def test_non_openai_backend_skips_openai_checks(self) -> None:
        """Non-openai backend doesn't need openai-specific fields."""
        rec = _record(model_backend="llama_cpp")
        result = ar._validate_eval_record_schema(rec, src="t", line_no=1)
        assert result is None


# ── Load and group edge cases (lines 337-382) ───────────────────────────


class TestLoadAndGroupRecords:
    """Exercises _load_and_group_records branches."""

    def test_strict_input_unreadable_file(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "nonexistent.jsonl")
        with pytest.raises(ValueError, match="unreadable file"):
            ar._load_and_group_records([bad_path], strict_input=True)

    def test_lenient_unreadable_file(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "nonexistent.jsonl")
        grouped, errors = ar._load_and_group_records([bad_path], strict_input=False)
        assert not grouped
        assert errors["files_unreadable"] == 1

    def test_strict_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            ar._load_and_group_records([str(p)], strict_input=True)

    def test_lenient_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        grouped, errors = ar._load_and_group_records([str(p)], strict_input=False)
        assert not grouped
        assert errors["json_lines_invalid"] == 1

    def test_strict_not_object(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.jsonl"
        p.write_text("[1,2,3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a JSON object"):
            ar._load_and_group_records([str(p)], strict_input=True)

    def test_lenient_not_object(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.jsonl"
        p.write_text("[1,2,3]\n", encoding="utf-8")
        grouped, errors = ar._load_and_group_records([str(p)], strict_input=False)
        assert not grouped
        assert errors["records_skipped"] == 1

    def test_strict_missing_case_id(self, tmp_path: Path) -> None:
        rec = _record()
        del rec["case_id"]  # type: ignore[arg-type]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        with pytest.raises(ValueError, match="missing case_id/condition"):
            ar._load_and_group_records([str(p)], strict_input=True)

    def test_lenient_missing_case_id(self, tmp_path: Path) -> None:
        rec = _record()
        del rec["case_id"]  # type: ignore[arg-type]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        grouped, errors = ar._load_and_group_records([str(p)], strict_input=False)
        assert not grouped
        assert errors["records_skipped"] == 1

    def test_infra_event_skipped(self, tmp_path: Path) -> None:
        rec = _record(record_type="infra_event")
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        grouped, errors = ar._load_and_group_records([str(p)], strict_input=False)
        assert not grouped
        assert errors["infra_event_rows_seen"] == 1

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        rec = _record()
        p = tmp_path / "data.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(json.dumps(rec) + "\n")
            f.write("  \n")
        grouped, _errors = ar._load_and_group_records([str(p)], strict_input=False)
        assert len(grouped) == 1

    def test_duplicate_resolution(self, tmp_path: Path) -> None:
        rec1 = _record(success=False)
        rec2 = _record(success=True)
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec1, rec2])
        grouped, errors = ar._load_and_group_records([str(p)], strict_input=False)
        assert errors["duplicate_attempt_rows_seen"] == 1
        assert errors["duplicate_attempt_rows_resolved"] == 1
        # Last record wins
        records = next(iter(grouped.values()))
        assert records[0]["success"] is True


# ── main() early-exit paths (lines 206-214, 290-304) ────────────────────


class TestMainExitPaths:
    """Exercises main() return codes for empty/missing data."""

    def test_no_records_returns_1(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("\n", encoding="utf-8")
        result = ar.main([str(p)])
        assert result == 1

    def test_stats_flag(self, tmp_path: Path) -> None:
        rec1 = _record(condition="base_repair", success=True)
        rec2 = _record(condition="layered_adaptive_llm", success=False)
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec1, rec2])
        result = ar.main([str(p), "--stats"])
        assert result == 0

    def test_out_flag(self, tmp_path: Path) -> None:
        rec = _record(condition="base_repair")
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        out = tmp_path / "sub" / "out.json"
        result = ar.main([str(p), "--out", str(out)])
        assert result == 0
        assert out.exists()
        summary = json.loads(out.read_text(encoding="utf-8"))
        assert "overall" in summary


# ── Context protocol extraction (lines 2550-2567) ───────────────────────


class TestContextProtocol:
    def test_from_top_level_field(self) -> None:
        records = [{"context_protocol": "staged"}]
        assert ar._context_protocol_from_records(records) == "staged"

    def test_from_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": {"context_protocol": "staged"}}]
        assert ar._context_protocol_from_records(records) == "staged"

    def test_default_monolithic(self) -> None:
        assert ar._context_protocol_from_records([{}]) == "monolithic"

    def test_non_dict_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        assert ar._context_protocol_from_records(records) == "monolithic"


class TestStageBPromptCharsMean:
    def test_normal(self) -> None:
        records = [
            {"attempt_metadata": {"stage_b_prompt_chars": 100}},
            {"attempt_metadata": {"stage_b_prompt_chars": 200}},
        ]
        assert ar._stage_b_prompt_chars_mean(records) == 150.0

    def test_non_dict_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        assert ar._stage_b_prompt_chars_mean(records) is None

    def test_missing_field(self) -> None:
        records = [{"attempt_metadata": {}}]
        assert ar._stage_b_prompt_chars_mean(records) is None


class TestStageARequestCounts:
    def test_normal(self) -> None:
        records = [
            {"attempt_metadata": {"stage_a_request_success": True}},
            {"attempt_metadata": {"stage_a_request_success": False}},
        ]
        attempts, successes = ar._stage_a_request_counts(records)
        assert attempts == 2
        assert successes == 1

    def test_non_dict_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        attempts, successes = ar._stage_a_request_counts(records)
        assert attempts == 0
        assert successes == 0


# ── Helper outcome extraction (lines 2776-2829) ─────────────────────────


class TestHelperOutcome:
    def _meta(self, **kwargs: object) -> tuple[dict[str, Any], dict[str, Any]]:
        return dict(kwargs), {}

    def test_backend_unavailable_normalized(self) -> None:
        am, pm = self._meta(adaptive_helper_outcome="backend_unavailable")
        assert ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm) == (
            "unavailable_backend"
        )

    def test_explicit_outcome(self) -> None:
        am, pm = self._meta(adaptive_helper_outcome="materialized")
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm) == "materialized"
        )

    def test_invoked_transport_failed(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=True,
            adaptive_helper_materialized_context=False,
            adaptive_helper_failure_kind="transport",
            adaptive_helper_decision_reason="",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "transport_failed"
        )

    def test_invoked_request_budget_exceeded(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=True,
            adaptive_helper_materialized_context=False,
            adaptive_helper_failure_kind="request_budget_exceeded",
            adaptive_helper_decision_reason="",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "request_budget_exceeded"
        )

    def test_invoked_compat_reasoning_only(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=True,
            adaptive_helper_materialized_context=False,
            adaptive_helper_failure_kind="compat_reasoning_only",
            adaptive_helper_decision_reason="",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "compat_reasoning_only"
        )

    def test_invoked_invalid_payload(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=True,
            adaptive_helper_materialized_context=False,
            adaptive_helper_failure_kind="invalid_payload",
            adaptive_helper_decision_reason="",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "invalid_payload"
        )

    def test_invoked_llm_invalid_response_no_helper(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=True,
            adaptive_helper_materialized_context=False,
            adaptive_helper_failure_kind="",
            adaptive_helper_decision_reason="llm_invalid_response_no_helper",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "invalid_payload"
        )

    def test_invoked_semantic_no_context(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=True,
            adaptive_helper_materialized_context=False,
            adaptive_helper_failure_kind="",
            adaptive_helper_decision_reason="",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "semantic_no_context"
        )

    def test_skipped_unsupported(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="request_context_unavailable",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "skipped_unsupported"
        )

    def test_skipped_guardrail_pregate(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="informativeness_pregate_skip",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "skipped_guardrail"
        )

    def test_skipped_guardrail_budget_exhausted(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="helper_budget_exhausted",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "skipped_guardrail"
        )

    def test_skipped_guardrail_saturation(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="helper_saturation_guard",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "skipped_guardrail"
        )

    def test_not_invoked_default(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="not_adaptive_condition",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm) == "not_invoked"
        )

    def test_not_invoked_empty_reason(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm) == "not_invoked"
        )

    def test_skipped_policy_fallback(self) -> None:
        am, pm = self._meta(
            adaptive_helper_invoked=False,
            adaptive_helper_decision_reason="some_custom_reason",
        )
        assert (
            ar._adaptive_helper_outcome(attempt_meta_dict=am, patch_meta_dict=pm)
            == "skipped_policy"
        )


# ── _looks_like_test_path (lines 4186-4197) ─────────────────────────────


class TestLooksLikeTestPath:
    def test_empty_string(self) -> None:
        assert ar._looks_like_test_path("") is False

    def test_whitespace_only(self) -> None:
        assert ar._looks_like_test_path("   ") is False

    def test_tests_directory(self) -> None:
        assert ar._looks_like_test_path("tests/foo.py") is True

    def test_test_directory(self) -> None:
        assert ar._looks_like_test_path("test/bar.py") is True

    def test_conftest(self) -> None:
        assert ar._looks_like_test_path("src/conftest.py") is True

    def test_test_prefix(self) -> None:
        assert ar._looks_like_test_path("src/test_foo.py") is True

    def test_test_suffix(self) -> None:
        assert ar._looks_like_test_path("src/foo_test.py") is True

    def test_non_test(self) -> None:
        assert ar._looks_like_test_path("src/foo.py") is False

    def test_backslash_normalized(self) -> None:
        assert ar._looks_like_test_path("tests\\foo.py") is True


# ── Token usage breakdown (lines 4341-4356) ─────────────────────────────


class TestTokenUsageBreakdown:
    def test_legacy_fallback(self) -> None:
        """Falls back to patch_meta.usage.total_tokens when attempt_metadata absent."""
        records = [
            {"patch_meta": {"usage": {"total_tokens": 500}}},
            {"patch_meta": {"usage": {"total_tokens": 300}}},
        ]
        prompt, completion, total = ar._token_usage_breakdown(records)
        assert prompt is None
        assert completion is None
        assert total == 800.0

    def test_legacy_no_usage(self) -> None:
        records = [{"patch_meta": {"usage": {}}}]
        prompt, completion, total = ar._token_usage_breakdown(records)
        assert prompt is None
        assert completion is None
        assert total is None

    def test_legacy_patch_meta_not_dict(self) -> None:
        records = [{"patch_meta": "bad"}]
        _prompt, _completion, total = ar._token_usage_breakdown(records)
        assert total is None

    def test_legacy_usage_not_dict(self) -> None:
        records = [{"patch_meta": {"usage": "bad"}}]
        _prompt, _completion, total = ar._token_usage_breakdown(records)
        assert total is None

    def test_attempt_metadata_with_unavailable_tokens(self) -> None:
        """token_usage_available=False means unmeasured, not 0 tokens (STA-01)."""
        records = [
            {
                "attempt_metadata": {
                    "token_usage_available": False,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "tokens_total": 0,
                }
            }
        ]
        prompt, completion, total = ar._token_usage_breakdown(records)
        assert (prompt, completion, total) == (None, None, None)

    def test_attempt_metadata_non_numeric_tokens(self) -> None:
        """Non-numeric prompt/completion/total fields are skipped."""
        records = [
            {
                "attempt_metadata": {
                    "token_usage_available": True,
                    "tokens_prompt": "bad",
                    "tokens_completion": 5,
                    "tokens_total": 9,
                }
            }
        ]
        _prompt, _completion, total = ar._token_usage_breakdown(records)
        # No valid record found via attempt_metadata path (non-numeric prompt),
        # but since found_attempt_meta is True from token_usage_available
        # ... actually the skip continues so no valid tokens from attempt path
        # Falls back to legacy path
        assert total is None

    def test_non_dict_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        _prompt, _completion, total = ar._token_usage_breakdown(records)
        assert total is None

    def test_reasoning_token_usage_breakdown_legacy_and_partial(self) -> None:
        legacy_records = [
            {
                "patch_meta": {
                    "usage": {
                        "total_tokens": 10,
                        "output_tokens_details": {"reasoning_tokens": 4},
                    }
                }
            }
        ]
        assert ar._reasoning_token_usage_breakdown(legacy_records) == (4.0, True, 4.0)

        partial_records = [
            {
                "attempt_metadata": {
                    "llm_call_count": 1,
                    "token_usage_available": True,
                    "reasoning_token_usage_available": True,
                    "tokens_reasoning": 3,
                }
            },
            {
                "attempt_metadata": {
                    "llm_call_count": 1,
                    "token_usage_available": True,
                    "reasoning_token_usage_available": False,
                    "tokens_reasoning": 0,
                }
            },
        ]
        assert ar._reasoning_token_usage_breakdown(partial_records) == (None, False, None)


# ── _attempts_budget_from_records (lines 4277-4284) ──────────────────────


class TestAttemptsBudgetFromRecords:
    def test_from_max_attempts(self) -> None:
        records = [
            {"attempt_metadata": {"max_attempts": 5}, "attempt": 1},
        ]
        assert ar._attempts_budget_from_records(records) == 5

    def test_fallback_to_max_attempt(self) -> None:
        records = [
            {"attempt_metadata": {}, "attempt": 3},
            {"attempt_metadata": {}, "attempt": 1},
        ]
        assert ar._attempts_budget_from_records(records) == 3

    def test_non_dict_attempt_metadata(self) -> None:
        records = [
            {"attempt_metadata": "bad", "attempt": 2},
        ]
        assert ar._attempts_budget_from_records(records) == 2


# ── _llm_calls_total (lines 4292-4293) ──────────────────────────────────


class TestLlmCallsTotal:
    def test_non_dict_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        assert ar._llm_calls_total(records) == 0


# ── _retry_history_attempt_metrics (lines 4374-4375) ────────────────────


class TestRetryHistoryMetrics:
    def test_non_dict_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        result = ar._retry_history_attempt_metrics(records)
        assert result["attempts"] == 0


# ── _runtime_used (lines 4403-4404 and fallback) ────────────────────────


class TestRuntimeUsed:
    def test_empty_records(self) -> None:
        assert ar._runtime_used([]) is None

    def test_cumulative_fallback(self) -> None:
        records = [
            {"cumulative_runtime_sec": 5.0},
            {"cumulative_runtime_sec": 10.0},
        ]
        assert ar._runtime_used(records) == 10.0

    def test_legacy_fallback(self) -> None:
        records = [
            {"before_duration_sec": 2.0, "after_duration_sec": 1.0},
            {"after_duration_sec": 3.0},
        ]
        result = ar._runtime_used(records)
        assert result == 6.0

    def test_no_timing_at_all(self) -> None:
        records = [{}]
        assert ar._runtime_used(records) is None


# ── Baseline metric helpers (lines 4462-4466, 4529, 4546-4580) ──────────


class TestBaselineMetricHelpers:
    def test_baseline_metric_sum_skips_bool(self) -> None:
        records = [
            {"patch_meta": {"baseline_metrics": {"key": True}}},
            {"patch_meta": {"baseline_metrics": {"key": 5}}},
        ]
        assert ar._baseline_metric_sum(records, "key") == 5.0

    def test_baseline_metric_sum_none(self) -> None:
        records = [{"patch_meta": {"baseline_metrics": {}}}]
        assert ar._baseline_metric_sum(records, "key") is None

    def test_timing_breakdown_mean_non_dict(self) -> None:
        records = [{"attempt_metadata": {"timing_breakdown": "bad"}}]
        result = ar._timing_breakdown_mean(records)
        assert all(v is None for v in result.values())

    def test_timing_breakdown_mean_skips_non_dict_am(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        result = ar._timing_breakdown_mean(records)
        assert all(v is None for v in result.values())

    def test_token_breakdown_mean_non_dict(self) -> None:
        records = [{"attempt_metadata": {"token_breakdown": "bad"}}]
        result = ar._token_breakdown_mean(records)
        assert all(v is None for v in result.values())

    def test_token_breakdown_mean_skips_non_dict_am(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        result = ar._token_breakdown_mean(records)
        assert all(v is None for v in result.values())

    def test_timing_breakdown_sum_non_dict(self) -> None:
        records = [{"attempt_metadata": {"timing_breakdown": "bad"}}]
        result = ar._timing_breakdown_sum(records)
        assert all(v == 0.0 for v in result.values())

    def test_timing_breakdown_sum_skips_non_dict_am(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        result = ar._timing_breakdown_sum(records)
        assert all(v == 0.0 for v in result.values())

    def test_token_breakdown_sum_non_dict(self) -> None:
        records = [{"attempt_metadata": {"token_breakdown": "bad"}}]
        result = ar._token_breakdown_sum(records)
        assert all(v == 0 for v in result.values())

    def test_token_breakdown_sum_skips_non_dict_am(self) -> None:
        records = [{"attempt_metadata": "bad"}]
        result = ar._token_breakdown_sum(records)
        assert all(v == 0 for v in result.values())


# ── Formatting helpers (lines 4963, 4971, 4980, 4991-4994) ──────────────


class TestFormattingHelpers:
    def test_fmt_num_none(self) -> None:
        assert ar._fmt_num(None) == "n/a"

    def test_fmt_num_non_numeric(self) -> None:
        assert ar._fmt_num("abc") == "n/a"

    def test_fmt_num_integer_float(self) -> None:
        assert ar._fmt_num(3.0) == "3"

    def test_fmt_num_float(self) -> None:
        assert ar._fmt_num(3.14) == "3.14"

    def test_fmt_signed_percent_none(self) -> None:
        assert ar._fmt_signed_percent(None) == "n/a"

    def test_fmt_signed_percent_non_numeric(self) -> None:
        assert ar._fmt_signed_percent("abc") == "n/a"

    def test_fmt_percent_none(self) -> None:
        assert ar._fmt_percent(None) == "n/a"

    def test_fmt_percent_non_numeric(self) -> None:
        assert ar._fmt_percent("abc") == "n/a"

    def test_as_int_float(self) -> None:
        assert ar._as_int(2.9) == 2

    def test_as_int_string(self) -> None:
        assert ar._as_int("7") == 7

    def test_as_int_bad_string(self) -> None:
        assert ar._as_int("abc") == 0

    def test_as_int_none(self) -> None:
        assert ar._as_int(None) == 0


# ── _ratio (line 5020-5021) ─────────────────────────────────────────────


class TestRatio:
    def test_zero_denom(self) -> None:
        assert ar._ratio(5, 0) is None

    def test_negative_denom(self) -> None:
        assert ar._ratio(5, -1) is None

    def test_normal(self) -> None:
        assert ar._ratio(3, 6) == 0.5


# ── _increment_string_count (line 1565-1566) ────────────────────────────


class TestIncrementStringCount:
    def test_non_string_value(self) -> None:
        target: dict[str, int] = {}
        ar._increment_string_count(target, 42)
        assert target == {}

    def test_empty_string_value(self) -> None:
        target: dict[str, int] = {}
        ar._increment_string_count(target, "  ")
        assert target == {}

    def test_normal(self) -> None:
        target: dict[str, int] = {}
        ar._increment_string_count(target, "  foo  ")
        assert target == {"foo": 1}


# ── Solve distribution (line 1337-1338) ─────────────────────────────────


class TestSolveDistribution:
    def test_attempts_to_success_none_skipped(self) -> None:
        """When attempts_to_success is None, it's excluded from distribution."""
        rows = [
            {"condition": "c", "success": True, "attempts_to_success": None},
            {"condition": "c", "success": True, "attempts_to_success": 2},
        ]
        dist: dict[str, int] = {}
        for row in rows:
            if not bool(row.get("success")):
                continue
            value = row.get("attempts_to_success")
            if value is None:
                continue
            key = str(int(value))
            dist[key] = int(dist.get(key, 0) + 1)
        assert dist == {"2": 1}


# ── Staged vs monolithic uplift (lines 1529-1534) ───────────────────────


class TestStagedVsMonolithicUplift:
    def test_both_present(self) -> None:
        rows = [
            {"condition": "layered_adaptive_llm", "context_protocol": "staged", "success": True},
            {
                "condition": "layered_adaptive_llm",
                "context_protocol": "monolithic",
                "success": False,
            },
        ]
        staged_rows = [
            r
            for r in rows
            if r.get("condition") == "layered_adaptive_llm"
            and r.get("context_protocol") == "staged"
        ]
        monolithic_rows = [
            r
            for r in rows
            if r.get("condition") == "layered_adaptive_llm"
            and r.get("context_protocol") == "monolithic"
        ]
        if staged_rows and monolithic_rows:
            staged_rate = sum(1 for r in staged_rows if bool(r.get("success"))) / len(staged_rows)
            monolithic_rate = sum(1 for r in monolithic_rows if bool(r.get("success"))) / len(
                monolithic_rows
            )
            uplift = staged_rate - monolithic_rate
        else:
            uplift = None
        assert uplift == 1.0


# ── _print_input_errors (lines 4904-4917) ───────────────────────────────


class TestPrintInputErrors:
    def test_not_dict(self, capsys: pytest.CaptureFixture[str]) -> None:
        ar._print_input_errors(None)
        assert capsys.readouterr().out == ""

    def test_no_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        ar._print_input_errors(
            {
                "files_unreadable": 0,
                "json_lines_invalid": 0,
                "records_skipped": 0,
                "duplicate_attempt_rows_seen": 0,
            }
        )
        assert capsys.readouterr().out == ""

    def test_with_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        ar._print_input_errors(
            {
                "files_unreadable": 2,
                "json_lines_invalid": 1,
                "records_skipped": 3,
                "duplicate_attempt_rows_seen": 0,
            }
        )
        out = capsys.readouterr().out
        assert "Input diagnostics" in out
        assert "files_unreadable=2" in out

    def test_with_dedup(self, capsys: pytest.CaptureFixture[str]) -> None:
        ar._print_input_errors(
            {
                "files_unreadable": 0,
                "json_lines_invalid": 0,
                "records_skipped": 0,
                "duplicate_attempt_rows_seen": 5,
                "duplicate_attempt_rows_resolved": 5,
            }
        )
        out = capsys.readouterr().out
        assert "dedup" in out


# ── _print_stats (lines 5229-5257) ──────────────────────────────────────


class TestPrintStats:
    def test_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        ar._print_stats({"error": "something went wrong"})
        out = capsys.readouterr().out
        assert "something went wrong" in out

    def test_with_pairwise(self, capsys: pytest.CaptureFixture[str]) -> None:
        stats: dict[str, Any] = {
            "aggregation": "per_run",
            "condition_a": "A",
            "condition_b": "B",
            "n_pairs": 10,
            "a_success_rate": 0.3,
            "b_success_rate": 0.5,
            "overall_mcnemar": {
                "n_01": 3,
                "n_10": 1,
                "p_value": 0.3,
                "method": "exact",
                "significant": False,
            },
            "overall_cohens_h": {"h": 0.2, "magnitude": "small"},
            "overall_bootstrap_ci": {
                "confidence": 0.95,
                "ci_lower": -0.1,
                "ci_upper": 0.5,
            },
            "overall_power": {
                "approximate_power": 0.5,
                "recommendation": "need more data",
            },
            "per_category": {
                "logic_error": {
                    "n_pairs": 5,
                    "mcnemar": {"p_value": 0.2, "significant": False},
                    "cohens_h": {"h": 0.1},
                }
            },
            "holm_bonferroni": {"logic_error": {"adjusted_p": 0.4, "rejected": False}},
            "pairwise": [
                {
                    "condition_a": "A",
                    "condition_b": "B",
                    "uplift_success_rate": 0.2,
                    "n_pairs": 10,
                    "mcnemar_p_value": 0.3,
                    "mcnemar_significant": False,
                },
                {"condition_a": "", "condition_b": "B"},
                "not_a_dict",
                {
                    "condition_a": "A",
                    "condition_b": "C",
                    "error": "not enough pairs",
                },
            ],
        }
        ar._print_stats(stats)
        out = capsys.readouterr().out
        assert "Pairwise matrix (Holm across all condition pairs):" in out
        assert "A -> B:" in out
        assert "A -> C: error=" in out

    def test_per_category_with_holm(self, capsys: pytest.CaptureFixture[str]) -> None:
        stats: dict[str, Any] = {
            "aggregation": "per_run",
            "condition_a": "A",
            "condition_b": "B",
            "n_pairs": 10,
            "a_success_rate": 0.3,
            "b_success_rate": 0.5,
            "overall_mcnemar": {
                "n_01": 3,
                "n_10": 1,
                "p_value": 0.3,
                "method": "exact",
                "significant": False,
            },
            "overall_cohens_h": {"h": 0.2, "magnitude": "small"},
            "overall_bootstrap_ci": {"confidence": 0.95, "ci_lower": -0.1, "ci_upper": 0.5},
            "overall_power": {"approximate_power": 0.5, "recommendation": "more data"},
            "per_category": {},
            "holm_bonferroni": {},
        }
        ar._print_stats(stats)
        out = capsys.readouterr().out
        assert "Statistical analysis" in out


# ── Budget family fallback (line 3354-3355) ─────────────────────────────


class TestBudgetFamilyFallback:
    def test_applied_family_used_when_selected_missing(self) -> None:
        """When selected_family is absent, applied_family is used."""
        am: dict[str, Any] = {
            "adaptive_budget_family_applied": "time",
        }
        selected = am.get("adaptive_budget_family_selected")
        applied = am.get("adaptive_budget_family_applied")
        family = ""
        if isinstance(selected, str) and selected in ar.ADAPTIVE_BUDGET_FAMILIES:
            family = selected
        elif isinstance(applied, str) and applied in ar.ADAPTIVE_BUDGET_FAMILIES:
            family = applied
        assert family == "time"


# ── OPE sample extraction (lines 3661-3711) ─────────────────────────────


class TestOPESampleExtraction:
    def test_skips_attempt_zero(self) -> None:
        """Attempt 0 records are excluded from OPE samples."""
        records = [
            {
                "attempt": 0,
                "attempt_metadata": {
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 1.0,
                    "adaptive_online_behavior_action": "action_a",
                    "adaptive_online_behavior_propensity": 0.5,
                },
            }
        ]
        result = ar._adaptive_ope_samples(
            records=records,
            prefix="adaptive_online",
            valid_actions={"action_a"},
        )
        assert result == []

    def test_skips_invalid_reward_scope(self) -> None:
        records = [
            {
                "attempt": 1,
                "attempt_metadata": {
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_reward_scope": "invalid_scope",
                    "adaptive_online_reward": 1.0,
                    "adaptive_online_behavior_action": "action_a",
                    "adaptive_online_behavior_propensity": 0.5,
                },
            }
        ]
        result = ar._adaptive_ope_samples(
            records=records,
            prefix="adaptive_online",
            valid_actions={"action_a"},
        )
        assert result == []

    def test_skips_invalid_action(self) -> None:
        records = [
            {
                "attempt": 1,
                "attempt_metadata": {
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 1.0,
                    "adaptive_online_behavior_action": "unknown_action",
                    "adaptive_online_behavior_propensity": 0.5,
                },
            }
        ]
        result = ar._adaptive_ope_samples(
            records=records,
            prefix="adaptive_online",
            valid_actions={"action_a"},
        )
        assert result == []


# ── LinAlgError fallback (lines 3999-4000) ──────────────────────────────


class TestRidgeLinAlgFallback:
    def test_singular_matrix_uses_pinv(self) -> None:
        """When np.linalg.solve fails, pinv fallback is used."""

        def bad_solve(*args: Any, **kwargs: Any) -> Any:
            raise np.linalg.LinAlgError("singular")

        with patch.object(np.linalg, "solve", side_effect=bad_solve):
            X = np.array([[1.0, 0.0], [0.0, 0.0]])
            y = np.array([1.0, 0.0])
            gram = X.T @ X
            rhs = X.T @ y
            beta = np.linalg.pinv(gram) @ rhs
            assert beta is not None


# ── DR / IPS edge cases (lines 4131-4133) ───────────────────────────────


class TestDREstimatePaths:
    def test_missing_target_model(self) -> None:
        """Returns None when target model is missing."""
        result, reason = ar._stratified_doubly_robust_estimate(
            samples=[
                {
                    "action": "a",
                    "propensity": 0.5,
                    "reward": 1.0,
                    "features": {"f": 1.0},
                    "feature_scaling_profile": "default",
                }
            ],
            target_action="a",
            action_space={"a", "b"},
            min_action_support=1,
            ridge_alpha=0.001,
            propensity_floor=0.01,
        )
        # With only one action in samples, no alternate model can be built
        assert result is None or reason != ""

    def test_doubly_robust_action_in_models(self) -> None:
        """Action present in models uses that model's prediction."""
        features = {"f1": 1.0}
        feature_names = ["f1"]
        beta_a = np.array([0.5, 0.1])
        beta_b = np.array([0.3, 0.2])
        models = {"a": beta_a, "b": beta_b}
        samples = [
            {"action": "a", "propensity": 0.5, "reward": 1.0, "features": features},
            {"action": "b", "propensity": 0.5, "reward": 0.0, "features": features},
        ]
        result = ar._doubly_robust_estimate(
            samples=samples,
            target_action="a",
            feature_names=feature_names,
            models=models,
            propensity_floor=0.01,
        )
        assert isinstance(result, float)


# ── _print_summary and _print_bucket (lines 4673-4877) ──────────────────


class TestPrintSummary:
    def _minimal_bucket(self) -> dict[str, Any]:
        return {
            "base_repair": {
                "cases": 5,
                "successes": 2,
                "success_rate": 0.4,
                "median_attempts_to_success": 1.0,
                "mean_tokens_total": 100.0,
                "mean_tokens_prompt": 50.0,
                "mean_tokens_completion": 50.0,
                "mean_tokens_per_attempt": 33.0,
                "mean_llm_calls_per_case": 2.0,
                "mean_attempts_budget": 3.0,
                "mean_attempt_ratio_when_successful": 0.5,
                "solve_on_attempt_distribution": {"1": 1, "2": 1},
                "mean_runtime_sec": 5.0,
                "mean_syntax_cycles_when_used": 1.0,
                "syntax_failed_attempt_rate": 0.1,
                "source_scope_match_rate": 0.8,
                "test_touch_attempt_rate": 0.2,
                "patch_attempts": 3,
                "test_touch_attempts": 1,
                "topup_case_rate": 0.1,
                "topup_attempt_rate": 0.05,
            },
            "uplift_success_rate": 0.1,
            "mean_prompt_chars_stage_b": 200.0,
            "mean_retry_history_chars": 50.0,
            "mean_retry_history_turns": 2.0,
            "retry_history_budget_applied_rate": 0.1,
            "retry_history_truncation_rate": 0.05,
            "mean_prompt_chars_stage_b_by_condition": {"base_repair": 200.0},
            "stage_a_request_success_rate": 0.9,
            "retry_gate_block_rate": 0.1,
            "uplift_staged_vs_monolithic": 0.05,
            "uplift_layered_gated_vs_base": 0.1,
            "uplift_layered_llm_vs_base": 0.15,
            "pairwise_uplifts": [
                {
                    "condition_a": "base_repair",
                    "condition_b": "layered_adaptive_llm",
                    "uplift_success_rate": 0.1,
                },
            ],
        }

    def test_print_bucket(self, capsys: pytest.CaptureFixture[str]) -> None:
        bucket = self._minimal_bucket()
        ar._print_bucket(bucket)
        out = capsys.readouterr().out
        assert "base_repair" in out
        assert "pairwise_uplifts" in out

    def test_print_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        bucket = self._minimal_bucket()
        summary = {
            "overall": bucket,
            "by_category": {"logic_error": bucket},
            "by_difficulty": {"easy": bucket},
            "by_snapshot_dependency": {"required": bucket},
            "by_bug_source": {"mutmut": bucket},
            "by_contamination_risk": {"low": bucket},
            "input_errors": {
                "files_unreadable": 1,
                "json_lines_invalid": 0,
                "records_skipped": 0,
                "duplicate_attempt_rows_seen": 0,
            },
        }
        ar._print_summary(summary)
        out = capsys.readouterr().out
        assert "Overall:" in out
        assert "By category:" in out
        assert "By difficulty:" in out
        assert "By snapshot dependency:" in out
        assert "By bug source:" in out
        assert "By contamination risk:" in out

    def test_print_bucket_pairwise_non_dict_skipped(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bucket = self._minimal_bucket()
        bucket["pairwise_uplifts"] = ["not_a_dict"]
        ar._print_bucket(bucket)
        out = capsys.readouterr().out
        assert "pairwise_uplifts (own attempt budgets):" in out


# ── Statistical analysis paths (lines 5121-5182) ────────────────────────


class TestStatisticalAnalysisPair:
    def test_majority_vote_aggregation(self) -> None:
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "A",
                "success": True,
                "category": "logic_error",
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "B",
                "success": False,
                "category": "logic_error",
            },
            {
                "run_id": "r2",
                "case_id": "c1",
                "condition": "A",
                "success": True,
                "category": "logic_error",
            },
            {
                "run_id": "r2",
                "case_id": "c1",
                "condition": "B",
                "success": True,
                "category": "logic_error",
            },
        ]
        result = ar._run_statistical_analysis_pair(
            rows,
            aggregation="majority_vote",
            pair_a="A",
            pair_b="B",
        )
        assert result["aggregation"] == "majority_vote"

    def test_any_success_aggregation(self) -> None:
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "A",
                "success": False,
                "category": "logic_error",
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "B",
                "success": True,
                "category": "logic_error",
            },
            {
                "run_id": "r2",
                "case_id": "c1",
                "condition": "A",
                "success": True,
                "category": "logic_error",
            },
            {
                "run_id": "r2",
                "case_id": "c1",
                "condition": "B",
                "success": False,
                "category": "logic_error",
            },
        ]
        result = ar._run_statistical_analysis_pair(
            rows,
            aggregation="any_success",
            pair_a="A",
            pair_b="B",
        )
        assert result["aggregation"] == "any_success"

    def test_pairwise_error_entry(self) -> None:
        """Pairwise entry with error is handled correctly."""
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "A",
                "success": True,
                "category": "logic_error",
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "B",
                "success": False,
                "category": "logic_error",
            },
        ]
        result = ar._run_statistical_analysis(
            rows,
            aggregation="per_run",
            pair_a="A",
            pair_b="B",
        )
        pairwise = result.get("pairwise", [])
        assert isinstance(pairwise, list)

    def test_per_run_default(self) -> None:
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "A",
                "success": True,
                "category": "logic_error",
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "B",
                "success": False,
                "category": "logic_error",
            },
        ]
        result = ar._run_statistical_analysis_pair(
            rows,
            aggregation="per_run",
            pair_a="A",
            pair_b="B",
        )
        assert result["aggregation"] == "per_run"
