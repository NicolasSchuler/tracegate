"""POL-18 / POL-13: one chronological event loader for bootstrap replay and dataset export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import _json as json
from evals.bootstrap_policy_state import _load_policy_events
from evals.export_policy_dataset import _load_events
from evals.policy_event_log import load_policy_events, non_finite_fields


def _event(case: str, created: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 7,
        "run_id": "r",
        "case_id": case,
        "condition": "layered_adaptive_llm",
        "attempt": 1,
        "decision_id": f"d-{case}",
        "created_at_utc": created,
        **extra,
    }


def _write(path: Path, events: list[dict[str, Any]]) -> str:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return str(path)


def test_events_are_chronological_regardless_of_file_order(tmp_path: Path) -> None:
    late = _write(tmp_path / "a.jsonl", [_event("c-late", "2026-10-04T10:00:05Z")])
    early = _write(
        tmp_path / "b.jsonl",
        [_event("z-early", "2026-10-04T10:00:01Z"), _event("a-mid", "2026-10-04T10:00:03Z")],
    )
    for paths in ([late, early], [early, late]):
        events, _ = load_policy_events(paths, strict_input=True)
        # Export used to sort by case id and bootstrap followed glob order.
        assert [event.payload["case_id"] for event in events] == ["z-early", "a-mid", "c-late"]


def test_same_second_ties_keep_log_order(tmp_path: Path) -> None:
    second = "2026-10-04T10:00:00Z"
    path = _write(
        tmp_path / "log.jsonl",
        [_event("zz", second), _event("aa", second), _event("mm", second)],
    )
    events, _ = load_policy_events([path], strict_input=True)
    assert [event.payload["case_id"] for event in events] == ["zz", "aa", "mm"]


def test_first_occurrence_wins_and_both_tools_agree(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dup.jsonl",
        [
            _event("c", "2026-10-04T10:00:00Z", reward=0.1),
            _event("c", "2026-10-04T10:00:09Z", reward=0.9),
        ],
    )
    exported, export_errors = _load_events([path], strict_input=False)
    replayed, replay_errors = _load_policy_events([path], strict_input=False)
    assert [event["reward"] for event in exported] == [0.1]
    assert [event.payload["reward"] for event in replayed] == [0.1]
    assert export_errors["duplicate_events_seen"] == replay_errors["duplicate_events_seen"] == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"reward": "inf"},
        {"propensity": "nan"},
        {"feature_values": {"bias": 1.0, "x": "-Infinity"}},
        {"scores": {"invoke_helper": "inf"}},
    ],
)
def test_non_finite_values_are_rejected(tmp_path: Path, extra: dict[str, Any]) -> None:
    path = _write(tmp_path / "bad.jsonl", [_event("c", "2026-10-04T10:00:00Z", **extra)])
    events, errors = _load_events([path], strict_input=False)
    assert events == []
    assert errors["records_non_finite"] == 1
    with pytest.raises(ValueError, match="non-finite"):
        _load_events([path], strict_input=True)


def test_unsupported_schema_is_counted_separately(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v3.jsonl", [{**_event("c", "2026-10-04T10:00:00Z"), "schema_version": 3}]
    )
    events, errors = _load_events([path], strict_input=False)
    assert events == []
    assert errors["records_unsupported_schema"] == 1
    # Bootstrap replay doesn't filter by schema version.
    replayed, _ = _load_policy_events([path], strict_input=False)
    assert len(replayed) == 1


def test_non_finite_fields_lists_offending_paths() -> None:
    payload = {"reward": 1.0, "propensity": float("inf"), "feature_values": {"a": float("nan")}}
    assert non_finite_fields(payload) == ["propensity", "feature_values.a"]
    assert non_finite_fields({"reward": True, "scores": {"x": "label"}}) == []
