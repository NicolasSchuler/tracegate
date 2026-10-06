"""Resume state of an eval run: settings fingerprint, existing results, JSONL tail repair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import _json as json
from evals.eval_evidence import _xxh3_128_jsonlike
from evals.eval_recording import _build_patcher_config, _resolve_llmdebug_snapshot_schema_version
from evals.eval_types import (
    EVAL_RECORD_TYPE_INFRA_EVENT,
    CaseSpec,
    Condition,
    _TrailingResultsTailIssue,
)
from evals.transport_failures import _record_is_infra_resume_eligible, _record_type


def _build_run_settings_fingerprint(
    *,
    args: argparse.Namespace,
    cases: list[CaseSpec],
    conditions: list[Condition],
) -> str:
    """Compute a stable xxh3_128 fingerprint of the run-level settings."""
    payload = {
        "cases_dir": str(args.cases_dir),
        "selected_case_ids": [case.case_id for case in cases],
        "conditions": [condition.value for condition in conditions],
        "patcher_config": _build_patcher_config(args=args, patcher_name=str(args.patcher)),
        "record_prompt": str(args.record_prompt),
        "record_evidence": str(args.record_evidence),
        "record_env": sorted(key.strip() for key in args.record_env if str(key).strip()),
        "record_deps": str(args.record_deps),
        # ORC-05: the seed drives bandit exploration even without --deterministic.
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "retry_context_mode": str(args.retry_context_mode),
        "retry_history_max_chars": int(args.retry_history_max_chars),
        "prompt_budget_retry_history_chars": int(args.prompt_budget_retry_history_chars),
        # CAP-20: the snapshot format (e.g. dict-key rendering) is part of the prompt evidence,
        # and the executor decides which llmdebug install writes it (4-2#8).
        "llmdebug_snapshot_schema_version": _resolve_llmdebug_snapshot_schema_version(args=args),
    }
    return _xxh3_128_jsonlike(payload)


def _load_existing_run_state(
    *,
    results_path: Path,
    run_id: str,
    max_attempts_per_unit: int,
) -> tuple[
    set[tuple[str, str]],
    dict[str, Any] | None,
    str | None,
    int,
    _TrailingResultsTailIssue | None,
]:
    """Scan an existing results JSONL to determine completed units.

    ORC-04: a results file written across an ``--allow-settings-drift`` resume holds more
    than one ``run_settings_fingerprint``. Each new generation declares the one it replaces
    in ``settings_drift_from_fingerprint``, so such a chain is accepted and the *latest*
    generation's fingerprint and metadata are returned; an unlinked change is still an error.

    Returns:
        Tuple of (completed_unit_keys, latest_run_metadata,
        settings_fingerprint, records_count, trailing_tail_issue).
    """
    if not results_path.exists():
        return set(), None, None, 0, None

    per_unit_stats: dict[tuple[str, str], dict[str, Any]] = {}
    latest_run_metadata: dict[str, Any] | None = None
    settings_fingerprint: str | None = None
    seen_fingerprints: set[str] = set()
    records_count = 0
    trailing_tail_issue: _TrailingResultsTailIssue | None = None

    def _consume_record(*, record: dict[str, Any], line_no: int) -> None:
        nonlocal latest_run_metadata, settings_fingerprint, records_count

        record_run_id = record.get("run_id")
        if not isinstance(record_run_id, str) or not record_run_id:
            raise ValueError(f"{results_path}:{line_no}: run_id must be a non-empty string")
        if record_run_id != run_id:
            raise ValueError(
                f"{results_path}:{line_no}: run_id mismatch ({record_run_id!r} != {run_id!r})"
            )

        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{results_path}:{line_no}: case_id must be a non-empty string")
        condition = record.get("condition")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"{results_path}:{line_no}: condition must be a non-empty string")
        record_type = _record_type(record)
        attempt = record.get("attempt")
        if not isinstance(attempt, int) or attempt < 0:
            raise ValueError(f"{results_path}:{line_no}: attempt must be a non-negative integer")
        success = record.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{results_path}:{line_no}: success must be a boolean")

        run_metadata = record.get("run_metadata")
        if not isinstance(run_metadata, dict):
            raise ValueError(f"{results_path}:{line_no}: run_metadata must be an object")
        if run_metadata.get("run_id") != run_id:
            raise ValueError(f"{results_path}:{line_no}: run_metadata.run_id must match run_id")

        current_fingerprint = run_metadata.get("run_settings_fingerprint")
        if current_fingerprint is not None and (
            not isinstance(current_fingerprint, str) or not current_fingerprint
        ):
            raise ValueError(
                f"{results_path}:{line_no}: run_settings_fingerprint must be a non-empty string"
            )
        if isinstance(current_fingerprint, str):
            if settings_fingerprint is None:
                settings_fingerprint = current_fingerprint
                seen_fingerprints.add(current_fingerprint)
                latest_run_metadata = dict(run_metadata)
            elif settings_fingerprint != current_fingerprint:
                drift_from = run_metadata.get("settings_drift_from_fingerprint")
                linked = (
                    isinstance(drift_from, str)
                    and drift_from in seen_fingerprints
                    and current_fingerprint not in seen_fingerprints
                )
                if not linked:
                    raise ValueError(
                        f"{results_path}:{line_no}: inconsistent run_settings_fingerprint values"
                    )
                settings_fingerprint = current_fingerprint
                seen_fingerprints.add(current_fingerprint)
                latest_run_metadata = dict(run_metadata)
        elif latest_run_metadata is None:
            latest_run_metadata = dict(run_metadata)

        unit_key = (case_id, condition)
        if record_type == EVAL_RECORD_TYPE_INFRA_EVENT:
            records_count += 1
            return
        if _record_is_infra_resume_eligible(record):
            records_count += 1
            return
        unit_stats = per_unit_stats.setdefault(
            unit_key,
            {"max_attempt": -1, "any_success": False},
        )
        unit_stats["max_attempt"] = max(int(unit_stats["max_attempt"]), attempt)
        if success:
            unit_stats["any_success"] = True
        records_count += 1

    def _parse_record_from_line_bytes(*, line_bytes: bytes, line_no: int) -> dict[str, Any]:
        try:
            line_text = line_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{results_path}:{line_no}: invalid UTF-8 ({type(exc).__name__}: {exc})"
            ) from exc
        try:
            record = json.loads(line_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{results_path}:{line_no}: invalid JSON ({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"{results_path}:{line_no}: record must be a JSON object")
        return record

    pending_line_no: int | None = None
    pending_line_start_offset: int = 0
    pending_line_bytes: bytes | None = None
    offset_bytes = 0
    with results_path.open("rb") as handle:
        for line_no, line_bytes in enumerate(handle, start=1):
            line_start_offset = offset_bytes
            offset_bytes += len(line_bytes)
            if not line_bytes.strip():
                continue
            if pending_line_no is not None and pending_line_bytes is not None:
                record = _parse_record_from_line_bytes(
                    line_bytes=pending_line_bytes,
                    line_no=pending_line_no,
                )
                _consume_record(record=record, line_no=pending_line_no)
            pending_line_no = line_no
            pending_line_start_offset = line_start_offset
            pending_line_bytes = line_bytes

    if pending_line_no is not None and pending_line_bytes is not None:
        try:
            trailing_record = _parse_record_from_line_bytes(
                line_bytes=pending_line_bytes,
                line_no=pending_line_no,
            )
        except ValueError as exc:
            error_text = str(exc)
            is_parse_error = "invalid JSON" in error_text or "invalid UTF-8" in error_text
            if not is_parse_error:
                raise
            trailing_tail_issue = _TrailingResultsTailIssue(
                line_no=pending_line_no,
                truncate_offset_bytes=pending_line_start_offset,
                raw_line_bytes=pending_line_bytes,
                parse_error=error_text,
            )
        else:
            _consume_record(record=trailing_record, line_no=pending_line_no)

    required_attempts = max(1, int(max_attempts_per_unit))
    completed_units = {
        unit_key
        for unit_key, stats in per_unit_stats.items()
        if bool(stats.get("any_success")) or int(stats.get("max_attempt", -1)) >= required_attempts
    }
    return (
        completed_units,
        latest_run_metadata,
        settings_fingerprint,
        records_count,
        trailing_tail_issue,
    )


def _repair_trailing_results_tail(
    *,
    results_path: Path,
    run_id: str,
    issue: _TrailingResultsTailIssue,
) -> tuple[Path, int]:
    """Truncate a corrupt trailing record and archive the removed bytes.

    Returns:
        Tuple of (sidecar_backup_path, removed_bytes_count).
    """
    original_size = int(results_path.stat().st_size)
    truncate_offset = int(issue.truncate_offset_bytes)
    if truncate_offset < 0 or truncate_offset > original_size:
        raise OSError(f"invalid truncate offset {truncate_offset} for file size {original_size}")
    with results_path.open("rb") as source:
        source.seek(truncate_offset)
        removed_tail_bytes = source.read()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sidecar_path = results_path.with_name(f"{results_path.name}.corrupt-tail-{timestamp}.log")
    suffix_index = 1
    while sidecar_path.exists():
        sidecar_path = results_path.with_name(
            f"{results_path.name}.corrupt-tail-{timestamp}.{suffix_index}.log"
        )
        suffix_index += 1

    header = (
        "# llmdebug eval trailing-tail repair backup\n"
        f"run_id={run_id}\n"
        f"results_path={results_path}\n"
        f"line_no={issue.line_no}\n"
        f"truncate_offset_bytes={truncate_offset}\n"
        f"original_size_bytes={original_size}\n"
        f"parse_error={issue.parse_error}\n\n"
    ).encode("utf-8", errors="strict")
    with sidecar_path.open("wb") as sidecar:
        sidecar.write(header)
        sidecar.write(removed_tail_bytes)

    with results_path.open("r+b") as handle:
        handle.truncate(truncate_offset)
        handle.flush()

    removed_bytes = original_size - truncate_offset
    return sidecar_path, removed_bytes
