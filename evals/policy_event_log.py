"""One loader for adaptive-policy event logs, shared by bootstrap replay and dataset export.

The two used to disagree (2026-10-04 review, POL-18): bootstrap replay moved a duplicate to
the end of the replay, export sorted by case id, and neither was chronological, so a replayed
state (decay depends on order) depended on glob order. Here events are ordered by
``(created_at_utc, run_id, source, line)``, and the first occurrence of a duplicate wins.
``created_at_utc`` has one-second resolution, so ties fall back to log order, which is the
order in which the controller updated its model (``decision_index`` is assigned at decision
time and can differ under concurrency).

Events with a non-finite reward, propensity, score or feature value are rejected (POL-13).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _json as json

_SCALAR_NUMERIC_FIELDS = (
    "reward",
    "propensity",
    "behavior_propensity",
    "family_propensity",
)
_MAPPING_NUMERIC_FIELDS = (
    "scores",
    "family_scores",
    "feature_values",
    "family_feature_values",
)


@dataclass(frozen=True)
class LoadedPolicyEvent:
    """One parsed policy event plus its source location."""

    payload: dict[str, Any]
    src: str
    line_no: int


def empty_input_errors() -> dict[str, int]:
    return {
        "files_unreadable": 0,
        "json_lines_invalid": 0,
        "records_skipped": 0,
        "records_unsupported_schema": 0,
        "records_non_finite": 0,
        "duplicate_events_seen": 0,
        "duplicate_events_resolved": 0,
    }


def event_key(payload: Mapping[str, Any]) -> tuple[str, str, str, int, str] | None:
    """Identity of an event: run, case, condition, attempt and decision id."""
    run_id = str(payload.get("run_id") or "").strip()
    case_id = str(payload.get("case_id") or "").strip()
    condition = str(payload.get("condition") or "").strip()
    attempt = _safe_int(payload.get("attempt"), default=-1)
    if not run_id or not case_id or not condition or attempt < 0:
        return None
    decision_id = str(payload.get("decision_id") or "").strip()
    if not decision_id:
        decision_id = str(payload.get("created_at_utc") or "").strip()
    if not decision_id:
        decision_id = "unknown"
    return run_id, case_id, condition, attempt, decision_id


def chronological_key(event: LoadedPolicyEvent) -> tuple[str, str, str, int]:
    """Replay order: creation time, then run, then position in the log."""
    payload = event.payload
    return (
        str(payload.get("created_at_utc") or ""),
        str(payload.get("run_id") or ""),
        event.src,
        event.line_no,
    )


def non_finite_fields(payload: Mapping[str, Any]) -> list[str]:
    """Numeric fields whose value is present but not a finite number."""
    bad = [name for name in _SCALAR_NUMERIC_FIELDS if _not_finite(payload.get(name))]
    for name in _MAPPING_NUMERIC_FIELDS:
        values = payload.get(name)
        if isinstance(values, Mapping):
            bad.extend(f"{name}.{key}" for key, value in values.items() if _not_finite(value))
    return bad


def load_policy_events(
    paths: Iterable[str],
    *,
    strict_input: bool,
    supported_versions: Collection[int] | None = None,
) -> tuple[list[LoadedPolicyEvent], dict[str, int]]:
    """Parse, validate, deduplicate and chronologically order policy events.

    Invalid lines are counted in the returned error counters, or raise ``ValueError`` with
    *strict_input*. *supported_versions* limits the accepted ``schema_version`` values.
    """
    input_errors = empty_input_errors()
    loaded: list[LoadedPolicyEvent] = []
    for src in paths:
        try:
            lines = Path(str(src)).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            input_errors["files_unreadable"] += 1
            if strict_input:
                raise ValueError(f"{src}: unable to read ({exc})") from exc
            continue
        for line_no, raw_line in enumerate(lines, start=1):
            event = _parse_line(
                raw_line,
                src=str(src),
                line_no=line_no,
                strict_input=strict_input,
                supported_versions=supported_versions,
                input_errors=input_errors,
            )
            if event is not None:
                loaded.append(event)
    loaded.sort(key=chronological_key)
    first_seen: dict[tuple[str, str, str, int, str], LoadedPolicyEvent] = {}
    for event in loaded:
        key = event_key(event.payload)
        if key is None:
            continue
        if key in first_seen:
            input_errors["duplicate_events_seen"] += 1
            continue
        first_seen[key] = event
    input_errors["duplicate_events_resolved"] = input_errors["duplicate_events_seen"]
    return list(first_seen.values()), input_errors


def _parse_line(
    raw_line: str,
    *,
    src: str,
    line_no: int,
    strict_input: bool,
    supported_versions: Collection[int] | None,
    input_errors: dict[str, int],
) -> LoadedPolicyEvent | None:
    line = raw_line.strip()
    if not line:
        return None
    where = f"{src}:{line_no}"
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        input_errors["json_lines_invalid"] += 1
        if strict_input:
            raise ValueError(f"{where}: invalid JSON ({exc})") from exc
        return None
    if not isinstance(payload, dict):
        input_errors["records_skipped"] += 1
        if strict_input:
            raise ValueError(f"{where}: JSON root is not an object")
        return None
    if supported_versions is not None:
        schema_version = _safe_int(payload.get("schema_version"), default=-1)
        if schema_version not in supported_versions:
            input_errors["records_skipped"] += 1
            input_errors["records_unsupported_schema"] += 1
            if strict_input:
                raise ValueError(
                    f"{where}: unsupported policy event schema_version={schema_version!r}, "
                    f"expected one of {sorted(supported_versions)!r}"
                )
            return None
    if event_key(payload) is None:
        input_errors["records_skipped"] += 1
        if strict_input:
            raise ValueError(f"{where}: missing run/case/condition/attempt")
        return None
    bad = non_finite_fields(payload)
    if bad:
        input_errors["records_skipped"] += 1
        input_errors["records_non_finite"] += 1
        if strict_input:
            raise ValueError(f"{where}: non-finite values in {', '.join(bad[:5])}")
        return None
    return LoadedPolicyEvent(payload=payload, src=src, line_no=line_no)


def _not_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
