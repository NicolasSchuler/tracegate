"""Load eval result JSONL files and validate records against the supported schemas."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evals import _json as json
from evals.analysis_numeric import _as_int, _expand_paths
from evals.case_outcome import resolve_duplicate_attempt
from evals.eval_types import EVAL_RECORD_SCHEMA_VERSION, normalize_condition_name

# v7 marks the 2026-10-04 review semantics (see evals/README.md, "Record schema versions").
SUPPORTED_EVAL_RECORD_SCHEMA_VERSIONS = frozenset({5, 6, EVAL_RECORD_SCHEMA_VERSION})


# Records from this version on use the xxh3_128 prompt/evidence hash field names.
_XXH3_HASH_FIELDS_SINCE_SCHEMA = 6


REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "run_id",
    "case_id",
    "condition",
    "attempt",
    "success",
    "category",
    "difficulty",
    "snapshot_dependency",
    "run_metadata",
    "attempt_metadata",
    "model_metadata",
)


VALID_RECORD_TYPES = {"attempt", "infra_event"}


REQUIRED_RUN_METADATA_FIELDS = (
    "run_id",
    "started_at_utc",
    "deterministic",
    "seed",
    "python_version",
    "platform",
    "llmdebug_version",
    "patcher_name",
    "patcher_config_fingerprint",
)


REQUIRED_ATTEMPT_METADATA_FIELDS = (
    "attempt",
    "max_attempts",
    "attempt_ordinal_label",
    "llm_call_count",
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "retry_history_budget_chars",
    "retry_history_budget_applied",
    "prompt_path",
    "evidence_path",
)


CURRENT_PROMPT_HASH_FIELD = "prompt_xxh3_128"


LEGACY_PROMPT_HASH_FIELD = "prompt_sha256"


CURRENT_EVIDENCE_HASH_FIELD = "evidence_xxh3_128"


LEGACY_EVIDENCE_HASH_FIELD = "evidence_sha256"


CURRENT_PATCH_DIFF_HASH_FIELD = "patch_diff_xxh3_128"


LEGACY_PATCH_DIFF_HASH_FIELD = "patch_diff_sha256"


REQUIRED_MODEL_METADATA_FIELDS = ("backend", "name")


REQUIRED_OPENAI_MODEL_METADATA_FIELDS = (
    "base_url",
    "model",
    "temperature",
    "max_tokens",
    "response_mode",
    "api_compat_seed_requested",
    "api_compat_seed_acknowledged",
)


def _prompt_hash_field_for_schema(schema_version: int) -> str:
    if schema_version >= _XXH3_HASH_FIELDS_SINCE_SCHEMA:
        return CURRENT_PROMPT_HASH_FIELD
    return LEGACY_PROMPT_HASH_FIELD


def _evidence_hash_field_for_schema(schema_version: int) -> str:
    if schema_version >= _XXH3_HASH_FIELDS_SINCE_SCHEMA:
        return CURRENT_EVIDENCE_HASH_FIELD
    return LEGACY_EVIDENCE_HASH_FIELD


def _load_and_group_records(
    paths: list[str],
    *,
    strict_input: bool = False,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Load JSONL files and group records by (run_id, case_id, condition).

    Validates each record, deduplicates by attempt number, and tracks
    input errors (unreadable files, invalid JSON, missing fields).

    ``record_schema_versions`` lists every schema version read, so a consumer can warn
    when one analysis mixes pre-remediation records with current ones (1e#13).

    Returns:
        Tuple of (grouped records dict, input diagnostics dict).
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    schema_versions: set[int] = set()
    input_errors: dict[str, Any] = {
        "files_unreadable": 0,
        "json_lines_invalid": 0,
        "records_skipped": 0,
        "records_schema_invalid": 0,
        "infra_event_rows_seen": 0,
        "infra_event_rows_skipped": 0,
        "duplicate_attempt_rows_seen": 0,
        "duplicate_attempt_rows_resolved": 0,
        "duplicate_attempt_rows_conflicting": 0,
    }
    deduped_records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for src in _expand_paths(paths):
        src_path = Path(src)
        try:
            handle = src_path.open(encoding="utf-8")
        except OSError as exc:
            input_errors["files_unreadable"] += 1
            if strict_input:
                raise ValueError(f"{src}: unreadable file ({type(exc).__name__}: {exc})") from exc
            continue
        with handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    input_errors["json_lines_invalid"] += 1
                    if strict_input:
                        raise ValueError(f"{src}:{line_no}: invalid JSON ({exc})") from exc
                    continue
                if not isinstance(record, dict):
                    input_errors["records_skipped"] += 1
                    if strict_input:
                        raise ValueError(f"{src}:{line_no}: record is not a JSON object")
                    continue
                run_id = str(record.get("run_id") or src)
                case_id = str(record.get("case_id") or "")
                condition = normalize_condition_name(record.get("condition"))
                if not case_id or not condition:
                    input_errors["records_skipped"] += 1
                    if strict_input:
                        raise ValueError(f"{src}:{line_no}: missing case_id/condition")
                    continue
                try:
                    _validate_eval_record_schema(record, src=src, line_no=line_no)
                except ValueError:
                    # STA-25: --strict-input now governs schema errors too.
                    input_errors["records_schema_invalid"] += 1
                    if strict_input:
                        raise
                    continue
                schema_versions.add(_as_int(record.get("schema_version")))
                record_type = str(record.get("record_type") or "attempt").strip() or "attempt"
                if record_type == "infra_event":
                    input_errors["infra_event_rows_seen"] += 1
                    input_errors["infra_event_rows_skipped"] += 1
                    continue
                attempt = _as_int(record.get("attempt"))
                dedup_key = (run_id, case_id, condition, attempt)
                previous = deduped_records.get(dedup_key)
                if previous is not None:
                    input_errors["duplicate_attempt_rows_seen"] += 1
                    # DAT-24: one resolver for this reader and case_outcome.scored_attempts.
                    resolution = resolve_duplicate_attempt(previous, record)
                    if resolution.conflicting:
                        input_errors["duplicate_attempt_rows_conflicting"] += 1
                        if strict_input:
                            raise ValueError(
                                f"{src}:{line_no}: conflicting duplicate of "
                                f"{case_id}/{condition} attempt {attempt}: "
                                f"success={bool(previous.get('success'))} and "
                                f"success={bool(record.get('success'))} with equal priority"
                            )
                    if resolution.keep is previous:
                        continue
                if record.get("condition") != condition:
                    record = {**record, "condition": condition}
                deduped_records[dedup_key] = record
    input_errors["duplicate_attempt_rows_resolved"] = input_errors["duplicate_attempt_rows_seen"]
    input_errors["record_schema_versions"] = sorted(schema_versions)
    for run_id, case_id, condition, _attempt in sorted(deduped_records):
        grouped[(run_id, case_id, condition)].append(
            deduped_records[(run_id, case_id, condition, _attempt)]
        )
    return grouped, input_errors


def _validate_eval_record_schema(record: dict[str, Any], *, src: str, line_no: int) -> None:
    """Validate a single eval record against the supported schemas.

    Raises ValueError with a descriptive message on any constraint
    violation, including missing fields, wrong types, and cross-field
    consistency checks.
    """
    for key in REQUIRED_TOP_LEVEL_FIELDS:
        if key not in record:
            raise ValueError(f"{src}:{line_no}: missing required field '{key}'")

    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_EVAL_RECORD_SCHEMA_VERSIONS:
        raise ValueError(
            f"{src}:{line_no}: unsupported schema_version={schema_version!r}, "
            f"expected one of {sorted(SUPPORTED_EVAL_RECORD_SCHEMA_VERSIONS)}"
        )
    schema_version_int = int(schema_version)

    if not isinstance(record.get("run_id"), str) or not str(record.get("run_id")):
        raise ValueError(f"{src}:{line_no}: run_id must be a non-empty string")
    if not isinstance(record.get("case_id"), str) or not str(record.get("case_id")):
        raise ValueError(f"{src}:{line_no}: case_id must be a non-empty string")
    if not isinstance(record.get("condition"), str) or not str(record.get("condition")):
        raise ValueError(f"{src}:{line_no}: condition must be a non-empty string")
    record_type = str(record.get("record_type") or "attempt").strip() or "attempt"
    if record_type not in VALID_RECORD_TYPES:
        raise ValueError(
            f"{src}:{line_no}: record_type must be one of {sorted(VALID_RECORD_TYPES)}"
        )
    if not isinstance(record.get("success"), bool):
        raise ValueError(f"{src}:{line_no}: success must be a boolean")

    attempt = record.get("attempt")
    if not isinstance(attempt, int) or attempt < 0:
        raise ValueError(f"{src}:{line_no}: attempt must be a non-negative integer")
    attempt_wall_clock_sec = record.get("attempt_wall_clock_sec")
    if attempt_wall_clock_sec is not None and (
        not isinstance(attempt_wall_clock_sec, (int, float)) or attempt_wall_clock_sec < 0
    ):
        raise ValueError(f"{src}:{line_no}: attempt_wall_clock_sec must be a non-negative number")

    run_metadata = record.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise ValueError(f"{src}:{line_no}: run_metadata must be an object")
    for key in REQUIRED_RUN_METADATA_FIELDS:
        if key not in run_metadata:
            raise ValueError(f"{src}:{line_no}: run_metadata missing '{key}'")
    if run_metadata.get("run_id") != record.get("run_id"):
        raise ValueError(f"{src}:{line_no}: run_metadata.run_id must match run_id")
    if not isinstance(run_metadata.get("deterministic"), bool):
        raise ValueError(f"{src}:{line_no}: run_metadata.deterministic must be a boolean")
    if run_metadata.get("seed") is not None and not isinstance(run_metadata.get("seed"), int):
        raise ValueError(f"{src}:{line_no}: run_metadata.seed must be int|null")

    attempt_metadata = record.get("attempt_metadata")
    if not isinstance(attempt_metadata, dict):
        raise ValueError(f"{src}:{line_no}: attempt_metadata must be an object")
    for key in REQUIRED_ATTEMPT_METADATA_FIELDS:
        if key not in attempt_metadata:
            raise ValueError(f"{src}:{line_no}: attempt_metadata missing '{key}'")
    prompt_hash_field = _prompt_hash_field_for_schema(schema_version_int)
    evidence_hash_field = _evidence_hash_field_for_schema(schema_version_int)
    for key in (prompt_hash_field, evidence_hash_field):
        if key not in attempt_metadata:
            raise ValueError(f"{src}:{line_no}: attempt_metadata missing '{key}'")
    if attempt_metadata.get("attempt") != attempt:
        raise ValueError(f"{src}:{line_no}: attempt_metadata.attempt must match attempt")
    _validate_optional_string_field(
        attempt_metadata.get(prompt_hash_field),
        path=f"attempt_metadata.{prompt_hash_field}",
        src=src,
        line_no=line_no,
    )
    _validate_optional_string_field(
        attempt_metadata.get("prompt_path"),
        path="attempt_metadata.prompt_path",
        src=src,
        line_no=line_no,
    )
    _validate_optional_string_field(
        attempt_metadata.get(evidence_hash_field),
        path=f"attempt_metadata.{evidence_hash_field}",
        src=src,
        line_no=line_no,
    )
    _validate_optional_string_field(
        attempt_metadata.get("evidence_path"),
        path="attempt_metadata.evidence_path",
        src=src,
        line_no=line_no,
    )
    max_attempts = attempt_metadata.get("max_attempts")
    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.max_attempts must be a non-negative int"
        )
    ordinal = attempt_metadata.get("attempt_ordinal_label")
    if not isinstance(ordinal, str) or not ordinal:
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.attempt_ordinal_label must be a non-empty string"
        )
    llm_call_count = attempt_metadata.get("llm_call_count")
    if not isinstance(llm_call_count, int) or llm_call_count < 0:
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.llm_call_count must be a non-negative int"
        )
    token_usage_available = attempt_metadata.get("token_usage_available")
    if token_usage_available is not None and not isinstance(token_usage_available, bool):
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.token_usage_available must be bool|null"
        )
    reasoning_token_usage_available = attempt_metadata.get("reasoning_token_usage_available")
    if reasoning_token_usage_available is not None and not isinstance(
        reasoning_token_usage_available, bool
    ):
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.reasoning_token_usage_available must be bool|null"
        )
    for key in ("tokens_prompt", "tokens_completion", "tokens_total", "retry_history_budget_chars"):
        value = attempt_metadata.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{src}:{line_no}: attempt_metadata.{key} must be a non-negative int")
    reasoning_tokens = attempt_metadata.get("tokens_reasoning")
    if reasoning_tokens is not None and (
        not isinstance(reasoning_tokens, int) or reasoning_tokens < 0
    ):
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.tokens_reasoning must be a non-negative int|null"
        )
    if not isinstance(attempt_metadata.get("retry_history_budget_applied"), bool):
        raise ValueError(
            f"{src}:{line_no}: attempt_metadata.retry_history_budget_applied must be a boolean"
        )

    model_metadata = record.get("model_metadata")
    if not isinstance(model_metadata, dict):
        raise ValueError(f"{src}:{line_no}: model_metadata must be an object")
    for key in REQUIRED_MODEL_METADATA_FIELDS:
        if key not in model_metadata:
            raise ValueError(f"{src}:{line_no}: model_metadata missing '{key}'")
    backend = model_metadata.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError(f"{src}:{line_no}: model_metadata.backend must be a non-empty string")

    if backend == "openai":
        for key in REQUIRED_OPENAI_MODEL_METADATA_FIELDS:
            if key not in model_metadata:
                raise ValueError(
                    f"{src}:{line_no}: model_metadata missing '{key}' for openai backend"
                )
        requested_seed = model_metadata.get("api_compat_seed_requested")
        if requested_seed is not None and not isinstance(requested_seed, int):
            raise ValueError(
                f"{src}:{line_no}: model_metadata.api_compat_seed_requested must be int|null"
            )
        acknowledged = model_metadata.get("api_compat_seed_acknowledged")
        if not isinstance(acknowledged, bool):
            raise ValueError(
                f"{src}:{line_no}: model_metadata.api_compat_seed_acknowledged must be a boolean"
            )


def _validate_optional_string_field(value: Any, *, path: str, src: str, line_no: int) -> None:
    """Raise ValueError if *value* is neither None nor a string."""
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{src}:{line_no}: {path} must be string|null")
