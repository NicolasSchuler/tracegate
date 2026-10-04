"""Tests for exporting policy events to a flat CSV dataset."""

from __future__ import annotations

import csv
import os
import stat

import pytest

from evals.adaptive_budget_online_policy import (
    EVENT_SCHEMA_VERSION as LIVE_BUDGET_EVENT_SCHEMA_VERSION,
)
from evals.adaptive_online_policy import (
    EVENT_SCHEMA_VERSION as LIVE_ONLINE_EVENT_SCHEMA_VERSION,
)
from evals.export_policy_dataset import (
    DECISION_MODEL_SCALAR_COLUMNS,
    POLICY_EVENT_SCHEMA_VERSION,
    SUPPORTED_BUDGET_EVENT_SCHEMA_VERSIONS,
    SUPPORTED_ONLINE_EVENT_SCHEMA_VERSIONS,
    SUPPORTED_POLICY_EVENT_SCHEMA_VERSIONS,
    _event_key,
    _expand_paths,
    _load_events,
    _optional_float_cell,
    _safe_float,
)
from evals.export_policy_dataset import (
    main as export_policy_dataset_main,
)
from llmdebug import _json as json


def test_export_policy_dataset_writes_flat_csv_and_dedupes(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    output_csv = tmp_path / "dataset.csv"
    events = [
        {
            "schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "run_id": "run_a",
            "case_id": "case_1",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "note": "no_patch_proposed",
            "decision_id": "d1",
            "decision_index": 1,
            "mode": "on",
            "algorithm": "linucb",
            "selected_action": "invoke_helper",
            "applied_action": "invoke_helper",
            "behavior_action": "invoke_helper",
            "default_action": "invoke_helper",
            "propensity": 0.8,
            "behavior_propensity": 0.8,
            "decision_strategy": "linucb_greedy",
            "feature_scaling_profile": "v1",
            "scores": {"skip_helper": 0.2, "invoke_helper": 0.5},
            "feature_values": {"bias": 1.0, "attempt_frac": 0.5},
            "reward_scope": "helper_local_v1",
            "reward": 0.2,
            "reward_success_hint": 0.0,
            "reward_progress_hint": 0.1,
            "reward_token_penalty": 0.5,
            "reward_latency_penalty": 0.3,
            "reward_stagnation_penalty": 0.0,
            "reward_helper_materialization_bonus": 0.05,
            "success": False,
            "local_llm_call_count": 1,
            "local_tokens_prompt": 200,
            "local_tokens_completion": 121,
            "local_tokens_total": 321,
            "local_wall_sec": 4.2,
            "override_reason": "",
            "model_updated": True,
            "update_reason": "on_policy_update",
            "created_at_utc": "2026-02-26T00:00:00Z",
        },
        {
            "schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "run_id": "run_a",
            "case_id": "case_1",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "note": "no_patch_proposed",
            "decision_id": "d1",
            "decision_index": 1,
            "mode": "on",
            "algorithm": "linucb",
            "selected_action": "invoke_helper",
            "applied_action": "invoke_helper",
            "behavior_action": "invoke_helper",
            "default_action": "invoke_helper",
            "propensity": 0.8,
            "behavior_propensity": 0.8,
            "decision_strategy": "linucb_greedy",
            "feature_scaling_profile": "v1",
            "scores": {"skip_helper": 0.2, "invoke_helper": 0.5},
            "feature_values": {"bias": 1.0, "attempt_frac": 0.5},
            "reward_scope": "helper_local_v1",
            "reward": 0.9,
            "reward_success_hint": 0.25,
            "reward_progress_hint": 0.0,
            "reward_token_penalty": 0.05,
            "reward_latency_penalty": 0.05,
            "reward_stagnation_penalty": 0.1,
            "reward_helper_materialization_bonus": 0.0,
            "success": True,
            "local_llm_call_count": 1,
            "local_tokens_prompt": 70,
            "local_tokens_completion": 30,
            "local_tokens_total": 100,
            "local_wall_sec": 1.0,
            "override_reason": "",
            "model_updated": True,
            "update_reason": "on_policy_update",
            "created_at_utc": "2026-02-26T00:00:01Z",
        },
        {
            "schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "run_id": "run_a",
            "case_id": "case_2",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "note": "tests_still_failing",
            "decision_id": "d2",
            "decision_index": 2,
            "mode": "shadow",
            "algorithm": "linucb",
            "selected_action": "skip_helper",
            "applied_action": "invoke_helper",
            "behavior_action": "invoke_helper",
            "default_action": "invoke_helper",
            "propensity": 0.0,
            "behavior_propensity": 1.0,
            "decision_strategy": "epsilon_uniform",
            "feature_scaling_profile": "v1",
            "scores": {"skip_helper": 0.4, "invoke_helper": 0.3},
            "feature_values": {"bias": 1.0, "attempt_frac": 1.0},
            "reward_scope": "helper_local_v1",
            "reward": -0.2,
            "reward_success_hint": 0.0,
            "reward_progress_hint": 0.0,
            "reward_token_penalty": 0.1,
            "reward_latency_penalty": 0.1,
            "reward_stagnation_penalty": 0.1,
            "reward_helper_materialization_bonus": 0.0,
            "success": False,
            "local_llm_call_count": 1,
            "local_tokens_prompt": 30,
            "local_tokens_completion": 20,
            "local_tokens_total": 50,
            "local_wall_sec": 2.0,
            "override_reason": "shadow_mode_baseline",
            "model_updated": False,
            "update_reason": "shadow_no_update",
            "created_at_utc": "2026-02-26T00:00:02Z",
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    exit_code = export_policy_dataset_main(
        [
            "--events",
            str(events_path),
            "--output-csv",
            str(output_csv),
            "--min-propensity",
            "0.1",
        ]
    )

    assert exit_code == 0
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    by_decision = {row["decision_id"]: row for row in rows}
    # POL-18: the chronologically first duplicate wins (this used to keep the last one).
    assert [row["decision_id"] for row in rows] == ["d1", "d2"]
    assert by_decision["d1"]["reward"] == "0.2"
    assert by_decision["d1"]["reward_scope"] == "helper_local_v1"
    assert by_decision["d1"]["feature_bias"] == "1.0"
    assert by_decision["d1"]["feature_attempt_frac"] == "0.5"
    assert by_decision["d1"]["importance_weight"] == "1.25"
    assert by_decision["d1"]["behavior_importance_weight"] == "1.25"
    assert by_decision["d1"]["feature_scaling_profile"] == "v1"
    assert by_decision["d1"]["reward_success_hint"] == "0.0"
    assert by_decision["d1"]["local_tokens_total"] == "321"
    assert by_decision["d1"]["local_wall_sec"] == "4.2"
    assert "tokens_total" not in by_decision["d1"]
    assert "attempt_wall_sec" not in by_decision["d1"]
    assert by_decision["d2"]["propensity_clipped"] == "0.1"
    assert by_decision["d2"]["importance_weight"] == "10.0"
    assert by_decision["d2"]["behavior_propensity"] == "1.0"
    assert by_decision["d2"]["behavior_importance_weight"] == "1.0"


def test_export_policy_dataset_strict_input_fails_fast(tmp_path) -> None:
    events_path = tmp_path / "events_invalid.jsonl"
    events_path.write_text('{"run_id":"r1"}\n{invalid-json\n', encoding="utf-8")

    exit_code = export_policy_dataset_main(
        [
            "--events",
            str(events_path),
            "--strict-input",
        ]
    )

    assert exit_code == 2


def test_export_policy_dataset_supports_budget_policy_events(tmp_path) -> None:
    events_path = tmp_path / "budget_events.jsonl"
    output_csv = tmp_path / "budget_dataset.csv"
    events = [
        {
            "schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "run_id": "run_budget",
            "case_id": "case_budget_1",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "note": "no_patch_proposed",
            "decision_id": "b1",
            "decision_index": 1,
            "mode": "on",
            "algorithm": "linucb",
            "architecture": "hierarchical_v1",
            "selected_action": "increase_budget",
            "applied_action": "increase_budget",
            "behavior_action": "increase_budget",
            "default_action": "keep_budget",
            "selected_family": "tokens",
            "applied_family": "tokens",
            "behavior_family": "tokens",
            "family_propensity": 0.5,
            "family_decision_strategy": "linucb_greedy",
            "selected_multiplier": 1.5,
            "applied_multiplier": 1.5,
            "default_multiplier": 1.0,
            "tier_index_before": 0,
            "tier_index_after": 1,
            "propensity": 0.6,
            "behavior_propensity": 0.6,
            "decision_strategy": "linucb_greedy",
            "feature_scaling_profile": "v1",
            "scores": {"keep_budget": 0.1, "increase_budget": 0.4},
            "feature_values": {"bias": 1.0, "attempt_frac": 0.5},
            "family_scores": {"time": 0.1, "tokens": 0.6, "context": 0.2},
            "family_feature_values": {"bias": 1.0, "budget_can_raise_tokens": 1.0},
            "reward_scope": "budget_local_v1",
            "reward": -0.3,
            "reward_success_hint": 0.0,
            "reward_progress_hint": 0.15,
            "reward_token_penalty": 0.2,
            "reward_latency_penalty": 0.1,
            "reward_stagnation_penalty": 0.0,
            "reward_budget_exceeded_penalty": 0.5,
            "request_budget_exceeded": True,
            "success": False,
            "local_llm_call_count": 2,
            "local_tokens_prompt": 80,
            "local_tokens_completion": 40,
            "local_tokens_total": 120,
            "local_wall_sec": 3.0,
            "override_reason": "",
            "gate_model_updated": True,
            "family_model_updated": True,
            "model_updated": True,
            "update_reason": "on_policy_update",
            "created_at_utc": "2026-03-04T00:00:00Z",
        }
    ]
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    exit_code = export_policy_dataset_main(
        [
            "--events",
            str(events_path),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["score_keep_budget"] == "0.1"
    assert row["score_increase_budget"] == "0.4"
    assert row["request_budget_exceeded"] == "1"
    assert row["reward_budget_exceeded_penalty"] == "0.5"
    assert row["applied_multiplier"] == "1.5"
    assert row["behavior_propensity"] == "0.6"
    assert row["architecture"] == "hierarchical_v1"
    assert row["selected_family"] == "tokens"
    assert row["family_propensity"] == "0.5"
    assert row["family_score_tokens"] == "0.6"
    assert row["family_feature_budget_can_raise_tokens"] == "1.0"
    assert row["family_model_updated"] == "1"
    assert row["feature_scaling_profile"] == "v1"
    assert row["reward_progress_hint"] == "0.15"
    assert row["local_tokens_total"] == "120"


def test_export_policy_dataset_rejects_legacy_event_schema(tmp_path) -> None:
    events_path = tmp_path / "legacy_events.jsonl"
    output_csv = tmp_path / "legacy_dataset.csv"
    events = [
        {
            "schema_version": 1,
            "run_id": "run_legacy",
            "case_id": "case_legacy_1",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "decision_id": "legacy_1",
            "selected_action": "skip_helper",
            "applied_action": "invoke_helper",
            "propensity": 0.25,
            "reward": 0.1,
            "scores": {"skip_helper": 0.4, "invoke_helper": 0.6},
            "feature_values": {"bias": 1.0},
            "created_at_utc": "2026-03-05T00:00:00Z",
        }
    ]
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    exit_code = export_policy_dataset_main(
        [
            "--events",
            str(events_path),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 1
    assert not output_csv.exists()


# ---------------------------------------------------------------------------
# Coverage gap: lines 55-56 — no files matched --events
# ---------------------------------------------------------------------------
def test_no_matching_events_returns_1(tmp_path, monkeypatch) -> None:
    """When _expand_paths returns empty, main() prints a message and returns 1."""
    monkeypatch.setattr("evals.export_policy_dataset._expand_paths", lambda _items: [])
    exit_code = export_policy_dataset_main(
        [
            "--events",
            str(tmp_path / "nonexistent_*.jsonl"),
            "--output-csv",
            str(tmp_path / "out.csv"),
        ]
    )
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Coverage gap: lines 102-106 — unreadable file (OSError), non-strict
# ---------------------------------------------------------------------------
def test_unreadable_file_skipped_in_non_strict_mode(tmp_path) -> None:
    """Unreadable files are silently skipped when strict_input=False."""
    bad_file = tmp_path / "unreadable.jsonl"
    bad_file.write_text("should not read", encoding="utf-8")
    bad_file.chmod(0o000)

    events, errors = _load_events([str(bad_file)], strict_input=False)

    # Restore permissions for cleanup
    bad_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert events == []
    assert errors["files_unreadable"] == 1


def test_unreadable_file_raises_in_strict_mode(tmp_path) -> None:
    """Unreadable files raise ValueError when strict_input=True."""
    bad_file = tmp_path / "unreadable.jsonl"
    bad_file.write_text("should not read", encoding="utf-8")
    bad_file.chmod(0o000)

    with pytest.raises(ValueError, match="unable to read"):
        _load_events([str(bad_file)], strict_input=True)

    bad_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# Coverage gap: lines 110, 113-117 — malformed JSON, non-strict and strict
# ---------------------------------------------------------------------------
def test_malformed_json_line_skipped_non_strict(tmp_path) -> None:
    """Malformed JSON lines are counted as errors but skipped in non-strict mode."""
    events_file = tmp_path / "bad.jsonl"
    events_file.write_text("{bad json\n", encoding="utf-8")

    events, errors = _load_events([str(events_file)], strict_input=False)

    assert events == []
    assert errors["json_lines_invalid"] == 1


def test_malformed_json_line_raises_in_strict(tmp_path) -> None:
    """Malformed JSON lines raise ValueError in strict mode."""
    events_file = tmp_path / "bad.jsonl"
    events_file.write_text("{bad json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        _load_events([str(events_file)], strict_input=True)


# ---------------------------------------------------------------------------
# Coverage gap: lines 119-122 — JSON root is not a dict
# ---------------------------------------------------------------------------
def test_non_dict_json_root_skipped_non_strict(tmp_path) -> None:
    """A JSON line whose root is a list (not a dict) is skipped."""
    events_file = tmp_path / "array.jsonl"
    events_file.write_text("[1, 2, 3]\n", encoding="utf-8")

    events, errors = _load_events([str(events_file)], strict_input=False)

    assert events == []
    assert errors["records_skipped"] == 1


def test_non_dict_json_root_raises_in_strict(tmp_path) -> None:
    """A non-dict JSON root raises ValueError in strict mode."""
    events_file = tmp_path / "array.jsonl"
    events_file.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not an object"):
        _load_events([str(events_file)], strict_input=True)


# ---------------------------------------------------------------------------
# Coverage gap: unsupported schema_version, strict
# ---------------------------------------------------------------------------
def test_unsupported_schema_version_raises_in_strict(tmp_path) -> None:
    events_file = tmp_path / "old.jsonl"
    events_file.write_text(
        json.dumps({"schema_version": 999, "run_id": "r"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported policy event schema_version"):
        _load_events([str(events_file)], strict_input=True)


# ---------------------------------------------------------------------------
# Coverage gap: lines 135-138 — missing event key fields, strict
# ---------------------------------------------------------------------------
def test_missing_event_key_skipped_non_strict(tmp_path) -> None:
    """An event without run_id/case_id/condition/attempt is skipped."""
    events_file = tmp_path / "nokey.jsonl"
    events_file.write_text(
        json.dumps({"schema_version": POLICY_EVENT_SCHEMA_VERSION}) + "\n",
        encoding="utf-8",
    )

    events, errors = _load_events([str(events_file)], strict_input=False)

    assert events == []
    assert errors["records_skipped"] == 1


def test_missing_event_key_raises_in_strict(tmp_path) -> None:
    """An event without required key fields raises ValueError in strict mode."""
    events_file = tmp_path / "nokey.jsonl"
    events_file.write_text(
        json.dumps({"schema_version": POLICY_EVENT_SCHEMA_VERSION}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing run/case/condition/attempt"):
        _load_events([str(events_file)], strict_input=True)


# ---------------------------------------------------------------------------
# Coverage gap: _event_key with various missing fields
# ---------------------------------------------------------------------------
def test_event_key_returns_none_for_empty_fields() -> None:
    """_event_key returns None when required fields are missing/empty."""
    assert _event_key({}) is None
    assert _event_key({"run_id": "r", "case_id": "", "condition": "c", "attempt": 1}) is None
    assert _event_key({"run_id": "r", "case_id": "c", "condition": "cond", "attempt": -1}) is None


def test_event_key_fallback_to_created_at_utc() -> None:
    """_event_key falls back to created_at_utc when decision_id is absent."""
    key = _event_key(
        {
            "run_id": "r",
            "case_id": "c",
            "condition": "cond",
            "attempt": 1,
            "created_at_utc": "2025-01-01",
        }
    )
    assert key is not None
    assert key[4] == "2025-01-01"


def test_event_key_falls_back_to_unknown() -> None:
    """_event_key uses 'unknown' when neither decision_id nor created_at_utc exist."""
    key = _event_key({"run_id": "r", "case_id": "c", "condition": "cond", "attempt": 1})
    assert key is not None
    assert key[4] == "unknown"


# ---------------------------------------------------------------------------
# Coverage gap: lines 153, 156, 158 — to_typed_dict conditional fields
# (covered via dedup: duplicate events resolve to last-writer-wins)
# ---------------------------------------------------------------------------
def test_duplicate_events_resolved_to_first(tmp_path) -> None:
    """Duplicate events (same key) are deduplicated; the first occurrence in chronological
    order wins (POL-18; this used to keep the last one)."""
    base = {
        "schema_version": POLICY_EVENT_SCHEMA_VERSION,
        "run_id": "r",
        "case_id": "c",
        "condition": "cond",
        "attempt": 1,
        "decision_id": "dup",
    }
    first = {**base, "reward": 0.1}
    second = {**base, "reward": 0.9}
    events_file = tmp_path / "dup.jsonl"
    events_file.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    events, errors = _load_events([str(events_file)], strict_input=False)

    assert len(events) == 1
    assert events[0]["reward"] == 0.1
    assert errors["duplicate_events_seen"] == 1
    assert errors["duplicate_events_resolved"] == 1


# ---------------------------------------------------------------------------
# Coverage gap: empty JSONL file → no valid policy events
# ---------------------------------------------------------------------------
def test_empty_events_file_returns_no_events(tmp_path) -> None:
    """An empty JSONL file (only blank lines) returns no events."""
    events_file = tmp_path / "empty.jsonl"
    events_file.write_text("\n\n", encoding="utf-8")

    events, _errors = _load_events([str(events_file)], strict_input=False)

    assert events == []


# ---------------------------------------------------------------------------
# Coverage gap: line 361 — _safe_float NaN returns default
# ---------------------------------------------------------------------------
def test_safe_float_nan_returns_default() -> None:
    """_safe_float returns the default when value is NaN."""
    assert _safe_float(float("nan"), default=42.0) == 42.0


def test_safe_float_non_numeric_returns_default() -> None:
    """_safe_float returns the default for non-numeric values."""
    assert _safe_float("not_a_number", default=1.5) == 1.5
    assert _safe_float(None, default=0.0) == 0.0


# ---------------------------------------------------------------------------
# Coverage gap: _expand_paths with non-matching globs
# ---------------------------------------------------------------------------
def test_expand_paths_non_matching_glob_kept_verbatim() -> None:
    """Non-matching globs are kept as-is (fallback to literal path)."""
    result = _expand_paths(["/no/such/path/*.jsonl"])
    assert result == ["/no/such/path/*.jsonl"]


# ---------------------------------------------------------------------------
# Decision-model columns and explicit per-kind schema acceptance
# ---------------------------------------------------------------------------
def test_supported_schema_versions_cover_live_controller_versions() -> None:
    """The explicit version sets must keep covering the shipped controllers."""
    assert LIVE_ONLINE_EVENT_SCHEMA_VERSION in SUPPORTED_ONLINE_EVENT_SCHEMA_VERSIONS
    assert LIVE_BUDGET_EVENT_SCHEMA_VERSION in SUPPORTED_BUDGET_EVENT_SCHEMA_VERSIONS
    # Helper logs from before the decision model (v4) stay readable.
    assert 4 in SUPPORTED_ONLINE_EVENT_SCHEMA_VERSIONS
    assert sorted(SUPPORTED_POLICY_EVENT_SCHEMA_VERSIONS) == [4, 5, 6, 7]  # POL-01, POL-21 bumps


def _decision_model_block() -> dict[str, object]:
    return {
        "role": "policy",
        "status": "ok",
        "profile": "evidence_v1",
        "model": "clef-flash",
        "revision": "abc123",
        "error": "",
        "action_probs": {"skip_helper": 0.25, "invoke_helper": 0.75},
        "top_action": "invoke_helper",
        "top_prob": 0.75,
        "margin": 0.5,
        "entropy": 0.56,
        "atomic_action": "frame_only",
        "atomic_reason": "missing_runtime_values",
        "traceback_sufficient": 0.3,
        "missing_information": {"runtime_values": 0.7, "none": 0.1},
        "fix_confidence": 0.4,
        "latency_ms": 123.5,
        "input_tokens": 2048,
        "state_sha256": "deadbeef",
        "state_chars": 5120,
        "truncated_sections": ["9_test_output_tail"],
        "effect": {"prior_beta": 0.25, "prior_adjustment": {"skip_helper": -0.1}},
    }


def _policy_event(**overrides: object) -> dict[str, object]:
    event = {
        "schema_version": POLICY_EVENT_SCHEMA_VERSION,
        "run_id": "run_dm",
        "case_id": "case_dm",
        "condition": "layered_adaptive_llm",
        "attempt": 1,
        "decision_id": "dm1",
        "decision_index": 1,
        "mode": "on",
        "algorithm": "linucb",
        "selected_action": "invoke_helper",
        "applied_action": "invoke_helper",
        "behavior_action": "invoke_helper",
        "propensity": 0.8,
        "behavior_propensity": 0.8,
        "decision_strategy": "decision_model_greedy",
        "feature_scaling_profile": "v1",
        "scores": {"skip_helper": 0.2, "invoke_helper": 0.5},
        "feature_values": {"bias": 1.0},
        "reward_scope": "helper_credit_v2",
        "reward": 0.3,
        "model_updated": True,
        "created_at_utc": "2026-10-04T00:00:00Z",
    }
    event.update(overrides)
    return event


def _export_rows(tmp_path, events: list[dict[str, object]]) -> list[dict[str, str]]:
    events_path = tmp_path / "events.jsonl"
    output_csv = tmp_path / "dataset.csv"
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    exit_code = export_policy_dataset_main(
        ["--events", str(events_path), "--output-csv", str(output_csv)]
    )
    assert exit_code == 0
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_flattens_decision_model_block_for_v5_events(tmp_path) -> None:
    rows = _export_rows(
        tmp_path,
        [_policy_event(schema_version=5, decision_model=_decision_model_block())],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["dm_role"] == "policy"
    assert row["dm_status"] == "ok"
    assert row["dm_profile"] == "evidence_v1"
    assert row["dm_model"] == "clef-flash"
    assert row["dm_top_action"] == "invoke_helper"
    assert row["dm_top_prob"] == "0.75"
    assert row["dm_margin"] == "0.5"
    assert row["dm_entropy"] == "0.56"
    assert row["dm_atomic_action"] == "frame_only"
    assert row["dm_traceback_sufficient"] == "0.3"
    assert row["dm_fix_confidence"] == "0.4"
    assert row["dm_latency_ms"] == "123.5"
    assert row["dm_input_tokens"] == "2048"
    assert row["dm_state_sha256"] == "deadbeef"
    assert row["dm_prior_beta"] == "0.25"
    assert row["dm_p_invoke_helper"] == "0.75"
    assert row["dm_p_skip_helper"] == "0.25"
    assert row["dm_missing_runtime_values"] == "0.7"
    assert row["dm_missing_none"] == "0.1"


def test_export_leaves_decision_model_columns_empty_for_v4_events(tmp_path) -> None:
    rows = _export_rows(
        tmp_path,
        [
            _policy_event(
                schema_version=5, decision_id="dm1", decision_model=_decision_model_block()
            ),
            _policy_event(schema_version=4, decision_id="legacy1", case_id="case_legacy"),
        ],
    )

    assert len(rows) == 2
    by_decision = {row["decision_id"]: row for row in rows}
    legacy = by_decision["legacy1"]
    # Union-expanded dm columns exist for every row, but stay empty without a block so
    # that "no advice" is distinguishable from a zero probability.
    for column in (*DECISION_MODEL_SCALAR_COLUMNS, "dm_p_invoke_helper", "dm_missing_none"):
        assert legacy[column] == "", column
    assert by_decision["dm1"]["dm_p_invoke_helper"] == "0.75"


def test_export_handles_decision_model_block_without_advice(tmp_path) -> None:
    rows = _export_rows(
        tmp_path,
        [
            _policy_event(
                decision_model={
                    "role": "shadow",
                    "status": "timeout",
                    "error": "read timeout",
                    "action_probs": {},
                    "missing_information": {},
                    "effect": {},
                }
            )
        ],
    )

    row = rows[0]
    assert row["dm_role"] == "shadow"
    assert row["dm_status"] == "timeout"
    assert row["dm_top_prob"] == "0.0"
    assert row["dm_input_tokens"] == "0"
    assert row["dm_traceback_sufficient"] == ""
    assert row["dm_fix_confidence"] == ""
    assert row["dm_prior_beta"] == ""


def test_optional_float_cell_renders_empty_for_unusable_values() -> None:
    """Optional dm scalars stay empty instead of silently becoming 0.0."""
    assert _optional_float_cell(None) == ""
    assert _optional_float_cell("not_a_number") == ""
    assert _optional_float_cell(float("nan")) == ""
    assert _optional_float_cell("0.25") == 0.25
