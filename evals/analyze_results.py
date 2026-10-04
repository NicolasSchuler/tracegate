"""Analyze and summarize llmdebug eval JSONL results.

Loads evaluation records, validates schema, derives per-case-condition
summary rows, and aggregates metrics across categories, difficulties,
and snapshot dependencies.  Supports adaptive helper, online-policy,
and budget-policy roll-ups with off-policy evaluation (OPE).  Optional
statistical significance testing via McNemar's test, Cohen's h, and
bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.linear_model import RidgeCV

from . import _json as json
from .adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    ACTION_SKIP_HELPER,
    ONLINE_POLICY_ACTIONS,
)
from .case_outcome import case_outcome, is_resume_eligible, scored_attempts
from .decision_model.policy import atomic_action_from_answers
from .decision_model.schema import AtomicThresholds, ParsedAnswers, get_profile
from .eval_types import (
    CONDITION_FAMILIES,
    CONDITION_FAMILY_EXTERNAL_BASELINE,
    EVAL_RECORD_SCHEMA_VERSION,
    condition_family,
    normalize_condition_name,
)
from .stats import BOOTSTRAP_METHOD, DEFAULT_BOOTSTRAP_SEED, format_p_value

logger = logging.getLogger(__name__)

# v7 marks the 2026-10-04 review semantics (see evals/README.md, "Record schema versions").
SUPPORTED_EVAL_RECORD_SCHEMA_VERSIONS = frozenset({5, 6, EVAL_RECORD_SCHEMA_VERSION})
# Records from this version on use the xxh3_128 prompt/evidence hash field names.
_XXH3_HASH_FIELDS_SINCE_SCHEMA = 6
ADAPTIVE_ONLINE_REWARD_SCOPE_LEGACY = "helper_local_v1"
ADAPTIVE_BUDGET_REWARD_SCOPE_LEGACY = "budget_local_v1"
ADAPTIVE_ONLINE_REWARD_SCOPE = "helper_credit_v2"
ADAPTIVE_BUDGET_REWARD_SCOPE = "budget_credit_v2"
ADAPTIVE_ONLINE_REWARD_SCOPES = frozenset(
    {ADAPTIVE_ONLINE_REWARD_SCOPE_LEGACY, ADAPTIVE_ONLINE_REWARD_SCOPE}
)
ADAPTIVE_BUDGET_REWARD_SCOPES = frozenset(
    {ADAPTIVE_BUDGET_REWARD_SCOPE_LEGACY, ADAPTIVE_BUDGET_REWARD_SCOPE}
)
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
ADAPTIVE_ONLINE_ACTIONS = set(ONLINE_POLICY_ACTIONS)
ADAPTIVE_BUDGET_ACTIONS = {"keep_budget", "increase_budget"}
ADAPTIVE_BUDGET_FAMILIES = frozenset({"time", "tokens", "context"})
ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS = (
    ("always_skip", "skip_helper"),
    ("always_frame_only", "frame_only"),
    ("always_file_only", "file_only"),
    ("always_invoke_helper", "invoke_helper"),
)
ADAPTIVE_ONLINE_TARGET_POLICIES = dict(ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS)
ADAPTIVE_ONLINE_TARGET_POLICY_ORDER = tuple(
    target_name for target_name, _target_action in ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS
)
ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS = (
    ("always_keep_budget", "keep_budget"),
    ("always_increase_budget", "increase_budget"),
)
ADAPTIVE_BUDGET_TARGET_POLICIES = dict(ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS)
ADAPTIVE_BUDGET_TARGET_POLICY_ORDER = tuple(
    target_name for target_name, _target_action in ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS
)
# Conservative default remains unchanged; now exposed as an analyzer CLI flag for reproducibility.
ADAPTIVE_OPE_MIN_ACTION_SUPPORT = 2
ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR = 1e-9
ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT = 0.005
ADAPTIVE_OPE_RIDGE_ALPHA = 1e-3
ADAPTIVE_OPE_RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_ABS_TOL = 0.05
ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_REL_TOL = 0.2
ADAPTIVE_OPE_SHADOW_POLICY_MODE = "shadow"
DECISION_MODEL_ADVICE_STATUS_OK = "ok"
DECISION_MODEL_TARGET_GREEDY = "decision_model_greedy"
DECISION_MODEL_TARGET_STOCHASTIC = "decision_model_stochastic"
DECISION_MODEL_TARGET_ATOMIC = "decision_model_atomic"
# Thresholds of the atomic profile, applied offline to aux answers logged by other profiles.
DECISION_MODEL_ATOMIC_THRESHOLDS = get_profile("evidence_atomic_v1").atomic or AtomicThresholds()
# Target name -> derivation kind for the per-sample decision-model policies.  These are only
# emitted when at least one sample carries usable advice, so runs without a decision model keep
# their historical OPE output.
DECISION_MODEL_TARGET_KINDS: tuple[tuple[str, str], ...] = (
    (DECISION_MODEL_TARGET_GREEDY, "greedy"),
    (DECISION_MODEL_TARGET_STOCHASTIC, "stochastic"),
    (DECISION_MODEL_TARGET_ATOMIC, "atomic"),
)
# Ties in the greedy target and the baseline fallback resolve in controller action order.
_ONLINE_ACTION_RANK = {action: index for index, action in enumerate(ONLINE_POLICY_ACTIONS)}
EXTERNAL_BASELINE_CONDITIONS = frozenset(
    name
    for name, family in CONDITION_FAMILIES.items()
    if family == CONDITION_FAMILY_EXTERNAL_BASELINE
)
# Minimum per-action sample count required for cross-validated alpha selection.
# Below this threshold we fall back to ADAPTIVE_OPE_RIDGE_ALPHA.
_RIDGECV_MIN_SAMPLES = 10


def _prompt_hash_field_for_schema(schema_version: int) -> str:
    if schema_version >= _XXH3_HASH_FIELDS_SINCE_SCHEMA:
        return CURRENT_PROMPT_HASH_FIELD
    return LEGACY_PROMPT_HASH_FIELD


def _evidence_hash_field_for_schema(schema_version: int) -> str:
    if schema_version >= _XXH3_HASH_FIELDS_SINCE_SCHEMA:
        return CURRENT_EVIDENCE_HASH_FIELD
    return LEGACY_EVIDENCE_HASH_FIELD


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for analyzing llmdebug eval JSONL results.

    Loads JSONL files, validates records, computes per-condition success
    rates and metrics, and prints a human-readable summary.  Optionally
    runs statistical significance tests and writes a JSON summary file.

    Args:
        argv: Command-line arguments.  Uses sys.argv when None.

    Returns:
        Exit code: 0 on success, 1 when no data found, 2 on input error.
    """
    parser = argparse.ArgumentParser(description="Analyze llmdebug eval JSONL results.")
    parser.add_argument("paths", nargs="+", help="One or more JSONL result files (globs ok).")
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write machine-readable summary JSON.",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="Fail on first unreadable input file or malformed JSONL record.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run statistical significance tests (McNemar, Cohen's h, bootstrap CI).",
    )
    parser.add_argument(
        "--aggregation",
        choices=["per_run", "majority_vote", "any_success"],
        default="per_run",
        help="Aggregation strategy for multi-run (k>1) analysis.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=(
            "Seed for every bootstrap interval (default: "
            f"{DEFAULT_BOOTSTRAP_SEED}); recorded in analysis_config."
        ),
    )
    parser.add_argument(
        "--stats-pair-a",
        default="base_repair",
        help="Condition name for A side of statistical pairing (default: base_repair).",
    )
    parser.add_argument(
        "--stats-pair-b",
        default="layered_adaptive_llm",
        help="Condition name for B side of statistical pairing (default: layered_adaptive_llm).",
    )
    parser.add_argument(
        "--adaptive-ope-min-action-support",
        type=int,
        default=ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
        help=(
            "Minimum behavior-policy samples required per action before DR is enabled (default: 2)."
        ),
    )
    parser.add_argument(
        "--adaptive-ope-propensity-floor",
        type=float,
        default=ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
        help=(
            "Analysis-only lower bound for adaptive OPE propensities in IPS/SNIPS/DR "
            f"(default: {ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT:g})."
        ),
    )
    args = parser.parse_args(argv)
    adaptive_ope_min_action_support = int(args.adaptive_ope_min_action_support)
    if adaptive_ope_min_action_support < 1:
        print("Input error: --adaptive-ope-min-action-support must be >= 1")
        return 2
    adaptive_ope_propensity_floor = float(args.adaptive_ope_propensity_floor)
    if not (0.0 < adaptive_ope_propensity_floor <= 1.0):
        print("Input error: --adaptive-ope-propensity-floor must be in (0, 1]")
        return 2

    try:
        grouped, input_errors = _load_and_group_records(args.paths, strict_input=args.strict_input)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2
    if not grouped:
        _print_input_errors(input_errors)
        print("No records found.")
        return 1

    rows = _derive_case_condition_rows(grouped)
    if not rows:
        print("No valid case-condition records found.")
        return 1
    records_by_case_condition = _records_by_case_condition(grouped)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_snapshot_dependency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bug_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_contamination_risk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_difficulty[row["difficulty"]].append(row)
        by_snapshot_dependency[row["snapshot_dependency"]].append(row)
        by_bug_source[row.get("bug_source") or "hand_crafted"].append(row)
        by_contamination_risk[row.get("contamination_risk") or "unknown"].append(row)

    summary = {
        "overall": _summarize_bucket(
            rows,
            records_by_case_condition=records_by_case_condition,
            adaptive_ope_min_action_support=adaptive_ope_min_action_support,
            adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
        ),
        "by_category": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_category.items())
        },
        "by_difficulty": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_difficulty.items())
        },
        "by_snapshot_dependency": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_snapshot_dependency.items())
        },
        "by_bug_source": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_bug_source.items())
        },
        "by_contamination_risk": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_contamination_risk.items())
        },
        "analysis_config": {
            "adaptive_ope_min_action_support": adaptive_ope_min_action_support,
            "adaptive_ope_propensity_floor": adaptive_ope_propensity_floor,
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_method": BOOTSTRAP_METHOD,
        },
        "input_errors": input_errors,
    }

    _print_summary(summary)

    if args.stats:
        stats_result = _run_statistical_analysis(
            rows,
            aggregation=args.aggregation,
            pair_a=str(args.stats_pair_a),
            pair_b=str(args.stats_pair_b),
            seed=int(args.bootstrap_seed),
        )
        summary["statistics"] = stats_result
        _print_stats(stats_result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote summary JSON: {out_path}")

    return 0


def _load_and_group_records(
    paths: list[str],
    *,
    strict_input: bool = False,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, int]]:
    """Load JSONL files and group records by (run_id, case_id, condition).

    Validates each record, deduplicates by attempt number, and tracks
    input errors (unreadable files, invalid JSON, missing fields).

    Returns:
        Tuple of (grouped records dict, input error counters dict).
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    input_errors = {
        "files_unreadable": 0,
        "json_lines_invalid": 0,
        "records_skipped": 0,
        "infra_event_rows_seen": 0,
        "infra_event_rows_skipped": 0,
        "duplicate_attempt_rows_seen": 0,
        "duplicate_attempt_rows_resolved": 0,
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
                _validate_eval_record_schema(record, src=src, line_no=line_no)
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
                    # A resumed attempt's real outcome beats its runtime-error row.
                    if is_resume_eligible(record) and not is_resume_eligible(previous):
                        continue
                if record.get("condition") != condition:
                    record = {**record, "condition": condition}
                deduped_records[dedup_key] = record
    input_errors["duplicate_attempt_rows_resolved"] = input_errors["duplicate_attempt_rows_seen"]
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


def _derive_case_condition_rows(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Derive one summary row per (run_id, case_id, condition) group.

    Each row aggregates attempts into success, token usage, runtime,
    adaptive helper, online-policy, budget-policy, and patch-verifier
    metrics used by downstream reporting and statistical analysis.
    """
    rows: list[dict[str, Any]] = []
    for (run_id, case_id, condition), records in sorted(grouped.items()):
        ordered = _dedupe_records_by_attempt(records)
        outcome = case_outcome(ordered)  # STA-19: the one definition of "case succeeded"
        success = outcome is not None and outcome.success
        attempts_executed = _attempts_executed(ordered)
        attempts_budget = _attempts_budget_from_records(ordered)
        attempts_to_success = _attempts_to_success(ordered)
        success_attempt_ratio = (
            (float(attempts_to_success) / float(attempts_budget))
            if attempts_to_success is not None and attempts_budget > 0
            else None
        )
        token_usage = _token_usage(ordered)
        token_prompt, token_completion, token_total = (
            token_usage.prompt,
            token_usage.completion,
            token_usage.total,
        )
        token_reasoning, reasoning_token_usage_available, reasoning_token_usage_per_attempt_mean = (
            _reasoning_token_usage_breakdown(ordered)
        )
        tokens = token_total
        # STA-09: divide by the attempts that reported tokens, not all executed attempts.
        token_usage_per_attempt_mean = (
            (token_total / float(token_usage.covered_attempts))
            if token_total is not None and token_usage.covered_attempts > 0
            else None
        )
        runtime = _runtime_used(ordered)
        llm_calls_total = _llm_calls_total(ordered)
        retry_history_stats = _retry_history_attempt_metrics(ordered)
        exemplar = ordered[0]
        context_protocol = _context_protocol_from_records(ordered)
        stage_b_prompt_chars = _stage_b_prompt_chars_mean(ordered)
        stage_a_attempts, stage_a_successes = _stage_a_request_counts(ordered)
        retry_gate_attempts, retry_gate_blocks = _retry_gate_counts(ordered)
        patch_attempts, test_touch_attempts = _test_touch_attempt_metrics(ordered)
        relaxed_apply_attempts = _patch_meta_flag_count(
            ordered, "patch_apply_mode", "relaxed_fallback"
        )
        test_surface_modified_attempts = _patch_meta_flag_count(
            ordered, "test_surface_modified", True
        )
        adaptive_metrics = _adaptive_helper_attempt_metrics(ordered)
        adaptive_online_metrics = _adaptive_online_attempt_metrics(ordered)
        adaptive_budget_policy_metrics = _adaptive_budget_policy_attempt_metrics(ordered)
        adaptive_topup_metrics = _adaptive_budget_topup_attempt_metrics(ordered)
        syntax_preflight_metrics = _syntax_preflight_attempt_metrics(ordered)
        patch_verifier_scope_metrics = _patch_verifier_source_scope_metrics(ordered)
        baseline_contract_version = _baseline_patch_meta_last_str(ordered, "contract_version")
        baseline_fidelity_level = _baseline_patch_meta_last_str(ordered, "fidelity_level")
        baseline_repair_scope_policy = _baseline_patch_meta_last_str(ordered, "repair_scope_policy")
        baseline_prompt_contract_kind = _baseline_patch_meta_last_str(
            ordered, "prompt_contract_kind"
        )
        baseline_hybrid_guidance_mode = _baseline_patch_meta_last_str(
            ordered, "baseline_hybrid_guidance_mode"
        )
        baseline_hybrid_guidance_chars = _baseline_patch_meta_last_int(
            ordered, "baseline_hybrid_guidance_chars"
        )
        baseline_hybrid_guidance_sections = _baseline_patch_meta_last_str_list(
            ordered, "baseline_hybrid_guidance_sections"
        )
        baseline_output_contract = _baseline_patch_meta_last_str(ordered, "output_contract")
        baseline_upstream_commit = _baseline_patch_meta_last_str(ordered, "upstream_commit")
        is_external_baseline = _is_external_baseline_condition(condition)
        # BAS-01 / D3: a case the baseline can't represent is not a baseline failure.
        baseline_supported = not any(
            isinstance(item.get("patch_meta"), dict)
            and item["patch_meta"].get("baseline_supported") is False
            for item in ordered
        )
        baseline_attempt_rollup = (
            _baseline_attempt_rollup(ordered) if is_external_baseline else {"attempt_count": 0}
        )
        baseline_teacher_duration_sec_total = (
            _baseline_metric_sum(ordered, "teacher_duration_sec_total")
            if is_external_baseline
            else None
        )
        baseline_learner_duration_sec_total = (
            _baseline_metric_sum(ordered, "learner_duration_sec_total")
            if is_external_baseline
            else None
        )
        baseline_postcheck_duration_sec_total = (
            _baseline_metric_sum(ordered, "postcheck_duration_sec_total")
            if is_external_baseline
            else None
        )
        baseline_learner_empty_response_count = (
            _baseline_metric_sum(ordered, "learner_empty_response_count")
            if is_external_baseline
            else None
        )
        baseline_attempts_with_postcheck_count = (
            _baseline_metric_sum(ordered, "attempts_with_postcheck_count")
            if is_external_baseline
            else None
        )
        baseline_direct_gen_passed = (
            _baseline_metric_last_bool(ordered, "direct_gen_passed")
            if is_external_baseline
            else None
        )
        baseline_debug_added_value = (
            _baseline_metric_last_bool(ordered, "debug_added_value")
            if is_external_baseline
            else None
        )
        baseline_method_instrumentation = (
            _baseline_metric_last_bool(ordered, "method_instrumentation")
            if is_external_baseline
            else None
        )
        baseline_method_two_step_repair = (
            _baseline_metric_last_bool(ordered, "method_two_step_repair")
            if is_external_baseline
            else None
        )
        baseline_stopped_reason = (
            _baseline_metric_last_str(ordered, "stopped_reason") if is_external_baseline else None
        )
        baseline_is_passing = (
            _baseline_metric_last_bool(ordered, "is_passing") if is_external_baseline else None
        )
        baseline_test_feedback_steps = (
            _baseline_metric_sum(ordered, "test_feedback_steps") if is_external_baseline else None
        )
        baseline_test_feedback_nonempty_steps = (
            _baseline_metric_sum(ordered, "test_feedback_nonempty_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_improvement_steps = (
            _baseline_metric_sum(ordered, "test_feedback_improvement_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_regression_steps = (
            _baseline_metric_sum(ordered, "test_feedback_regression_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_stagnation_steps = (
            _baseline_metric_sum(ordered, "test_feedback_stagnation_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_final_fail_count = (
            _baseline_metric_last_int(ordered, "test_feedback_final_fail_count")
            if is_external_baseline
            else None
        )
        baseline_terminal_timeout = (
            (_baseline_meta_last_str(ordered, "error") == "timeout") and not success
            if is_external_baseline
            else None
        )
        helper_invocation_attempts = int(adaptive_metrics["helper_invocation_attempts"])
        helper_materialized_attempts = int(adaptive_metrics["helper_materialized_attempts"])
        helper_low_yield_attempts = int(adaptive_metrics["helper_low_yield_attempts"])
        helper_transport_failure_attempts = int(
            adaptive_metrics["helper_transport_failure_attempts"]
        )
        helper_backend_unavailable_attempts = int(
            adaptive_metrics["helper_backend_unavailable_attempts"]
        )
        helper_compat_reasoning_only_attempts = int(
            adaptive_metrics["helper_compat_reasoning_only_attempts"]
        )
        helper_invalid_payload_attempts = int(adaptive_metrics["helper_invalid_payload_attempts"])
        helper_request_budget_exceeded_attempts = int(
            adaptive_metrics["helper_request_budget_exceeded_attempts"]
        )
        helper_retry_attempts = int(adaptive_metrics["helper_retry_attempts"])
        helper_transport_retry_attempts = int(adaptive_metrics["helper_transport_retry_attempts"])
        helper_effective_attempts = max(
            0,
            helper_invocation_attempts
            - helper_transport_failure_attempts
            - helper_backend_unavailable_attempts
            - helper_compat_reasoning_only_attempts,
        )
        helper_invoked_case = helper_invocation_attempts > 0
        helper_materialized_case = helper_materialized_attempts > 0
        helper_success_with_invocation = bool(success and helper_invoked_case)
        first_helper_attempt = adaptive_metrics["first_helper_attempt"]
        helper_rescued = bool(
            helper_success_with_invocation
            and isinstance(first_helper_attempt, int)
            and first_helper_attempt > 1
        )
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "condition": condition,
                "category": str(exemplar.get("category") or "unknown"),
                "difficulty": str(exemplar.get("difficulty") or "unknown"),
                "snapshot_dependency": str(exemplar.get("snapshot_dependency") or "unknown"),
                "bug_source": exemplar.get("bug_source"),
                "contamination_risk": exemplar.get("contamination_risk"),
                "success": success,
                "attempts_executed": attempts_executed,
                "attempts_budget": attempts_budget,
                "attempts_to_success": attempts_to_success,
                "success_attempt_ratio": success_attempt_ratio,
                "token_usage": tokens,
                "token_usage_prompt": token_prompt,
                "token_usage_completion": token_completion,
                "token_usage_total": token_total,
                "token_usage_reasoning": token_reasoning,
                "reasoning_token_usage_available": reasoning_token_usage_available,
                "token_usage_per_attempt_mean": token_usage_per_attempt_mean,
                "token_covered_attempts": token_usage.covered_attempts,
                "token_covered_fraction": _ratio(token_usage.covered_attempts, attempts_executed),
                "reasoning_token_usage_per_attempt_mean": (reasoning_token_usage_per_attempt_mean),
                "runtime_sec": runtime,
                "llm_calls_total": llm_calls_total,
                "timing_breakdown_sum": _timing_breakdown_sum(ordered),
                "token_breakdown_sum": _token_breakdown_sum(ordered),
                "context_protocol": context_protocol,
                "stage_b_prompt_chars_mean": stage_b_prompt_chars,
                "stage_a_request_attempts": stage_a_attempts,
                "stage_a_request_successes": stage_a_successes,
                "retry_gate_attempts": retry_gate_attempts,
                "retry_gate_blocks": retry_gate_blocks,
                "patch_attempts": patch_attempts,
                "baseline_supported": baseline_supported,
                "baseline_compatibility_reason": (
                    None
                    if baseline_supported
                    else _baseline_patch_meta_last_str(ordered, "compatibility_reason")
                ),
                "test_touch_attempts": test_touch_attempts,
                "test_touch_attempt_rate": (
                    (test_touch_attempts / patch_attempts) if patch_attempts > 0 else None
                ),
                "relaxed_apply_attempts": relaxed_apply_attempts,
                "test_surface_modified_attempts": test_surface_modified_attempts,
                "baseline_teacher_duration_sec_total": baseline_teacher_duration_sec_total,
                "baseline_learner_duration_sec_total": baseline_learner_duration_sec_total,
                "baseline_postcheck_duration_sec_total": baseline_postcheck_duration_sec_total,
                "baseline_contract_version": baseline_contract_version,
                "baseline_fidelity_level": baseline_fidelity_level,
                "baseline_repair_scope_policy": baseline_repair_scope_policy,
                "baseline_prompt_contract_kind": baseline_prompt_contract_kind,
                "baseline_hybrid_guidance_mode": baseline_hybrid_guidance_mode,
                "baseline_hybrid_guidance_chars": baseline_hybrid_guidance_chars,
                "baseline_hybrid_guidance_sections": baseline_hybrid_guidance_sections,
                "baseline_output_contract": baseline_output_contract,
                "baseline_upstream_commit": baseline_upstream_commit,
                "baseline_learner_empty_response_count": baseline_learner_empty_response_count,
                "baseline_attempts_with_postcheck_count": baseline_attempts_with_postcheck_count,
                "baseline_direct_gen_passed": baseline_direct_gen_passed,
                "baseline_debug_added_value": baseline_debug_added_value,
                "baseline_method_instrumentation": baseline_method_instrumentation,
                "baseline_method_two_step_repair": baseline_method_two_step_repair,
                "baseline_stopped_reason": baseline_stopped_reason,
                "baseline_is_passing": baseline_is_passing,
                "baseline_test_feedback_steps": baseline_test_feedback_steps,
                "baseline_test_feedback_nonempty_steps": baseline_test_feedback_nonempty_steps,
                "baseline_test_feedback_improvement_steps": (
                    baseline_test_feedback_improvement_steps
                ),
                "baseline_test_feedback_regression_steps": baseline_test_feedback_regression_steps,
                "baseline_test_feedback_stagnation_steps": baseline_test_feedback_stagnation_steps,
                "baseline_test_feedback_final_fail_count": (
                    baseline_test_feedback_final_fail_count
                ),
                "baseline_terminal_timeout": baseline_terminal_timeout,
                "baseline_attempt_summary_count": baseline_attempt_rollup.get("attempt_count"),
                "baseline_iteration_budget_exhausted_attempt_count": baseline_attempt_rollup.get(
                    "iteration_budget_exhausted_attempt_count"
                ),
                "baseline_no_write_buggy_py_attempt_count": baseline_attempt_rollup.get(
                    "no_write_buggy_py_attempt_count"
                ),
                "baseline_no_write_buggy_py_attempt_covered": baseline_attempt_rollup.get(
                    "no_write_buggy_py_attempt_covered"
                ),
                "baseline_no_test_attempt_count": baseline_attempt_rollup.get(
                    "no_test_attempt_count"
                ),
                "baseline_no_test_attempt_covered": baseline_attempt_rollup.get(
                    "no_test_attempt_covered"
                ),
                "baseline_subcall_count_total": baseline_attempt_rollup.get("subcall_count_total"),
                "baseline_subcall_count_covered": baseline_attempt_rollup.get(
                    "subcall_count_covered"
                ),
                "baseline_test_run_count_total": baseline_attempt_rollup.get(
                    "test_run_count_total"
                ),
                "baseline_test_run_count_covered": baseline_attempt_rollup.get(
                    "test_run_count_covered"
                ),
                "adaptive_mode": adaptive_metrics["adaptive_mode"],
                "helper_invocation_attempts": helper_invocation_attempts,
                "helper_invocation_attempt_rate": (
                    (helper_invocation_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_invoked_case": helper_invoked_case,
                "helper_materialized_attempts": helper_materialized_attempts,
                "helper_materialized_attempt_rate": (
                    (helper_materialized_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_materialized_case": helper_materialized_case,
                "helper_low_yield_attempts": helper_low_yield_attempts,
                "helper_low_yield_attempt_rate": (
                    (helper_low_yield_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_transport_failure_attempts": helper_transport_failure_attempts,
                "helper_transport_failure_attempt_rate": (
                    (helper_transport_failure_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_backend_unavailable_attempts": helper_backend_unavailable_attempts,
                "helper_backend_unavailable_attempt_rate": (
                    (helper_backend_unavailable_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_compat_reasoning_only_attempts": helper_compat_reasoning_only_attempts,
                "helper_compat_reasoning_only_attempt_rate": (
                    (helper_compat_reasoning_only_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_invalid_payload_attempts": helper_invalid_payload_attempts,
                "helper_invalid_payload_attempt_rate": (
                    (helper_invalid_payload_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_request_budget_exceeded_attempts": (
                    helper_request_budget_exceeded_attempts
                ),
                "helper_request_budget_exceeded_attempt_rate": (
                    (helper_request_budget_exceeded_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_retry_attempts": helper_retry_attempts,
                "helper_transport_retry_attempts": helper_transport_retry_attempts,
                "helper_effective_attempts": helper_effective_attempts,
                "helper_effective_success_rate": (
                    (helper_materialized_attempts / helper_effective_attempts)
                    if helper_effective_attempts > 0
                    else None
                ),
                "helper_success_with_invocation": helper_success_with_invocation,
                "helper_rescued": helper_rescued,
                "helper_first_attempt": first_helper_attempt,
                "helper_calls_used_final": adaptive_metrics["helper_calls_used_final"],
                "helper_decision_attempts": adaptive_metrics["helper_decision_attempts"],
                "helper_need_more_true_attempts": adaptive_metrics[
                    "helper_need_more_true_attempts"
                ],
                "helper_need_more_false_attempts": adaptive_metrics[
                    "helper_need_more_false_attempts"
                ],
                "helper_budget_exhausted_attempts": adaptive_metrics[
                    "helper_budget_exhausted_attempts"
                ],
                "helper_saturation_guard_attempts": adaptive_metrics[
                    "helper_saturation_guard_attempts"
                ],
                "adaptive_budget_topup_attempts": adaptive_topup_metrics["topup_attempts"],
                "adaptive_budget_topup_case": bool(adaptive_topup_metrics["topup_attempts"] > 0),
                "adaptive_online_mode": adaptive_online_metrics["adaptive_online_mode"],
                "adaptive_online_policy_algo": adaptive_online_metrics[
                    "adaptive_online_policy_algo"
                ],
                "adaptive_online_decisions": adaptive_online_metrics["decisions"],
                "adaptive_online_selected_action_counts": adaptive_online_metrics[
                    "selected_action_counts"
                ],
                "adaptive_online_applied_action_counts": adaptive_online_metrics[
                    "applied_action_counts"
                ],
                "adaptive_online_behavior_action_counts": adaptive_online_metrics[
                    "behavior_action_counts"
                ],
                "adaptive_online_selected_invoke_attempts": adaptive_online_metrics[
                    "selected_invoke_attempts"
                ],
                "adaptive_online_applied_invoke_attempts": adaptive_online_metrics[
                    "applied_invoke_attempts"
                ],
                "adaptive_online_override_attempts": adaptive_online_metrics["override_attempts"],
                "adaptive_online_updates": adaptive_online_metrics["updates"],
                "adaptive_online_reward_sum": adaptive_online_metrics["reward_sum"],
                "adaptive_online_reward_count": adaptive_online_metrics["reward_count"],
                "adaptive_online_reward_values": adaptive_online_metrics["reward_values"],
                "adaptive_online_propensity_sum": adaptive_online_metrics["propensity_sum"],
                "adaptive_online_propensity_count": adaptive_online_metrics["propensity_count"],
                "adaptive_online_local_tokens_total_sum": adaptive_online_metrics[
                    "local_tokens_total_sum"
                ],
                "adaptive_online_local_tokens_total_count": adaptive_online_metrics[
                    "local_tokens_total_count"
                ],
                "adaptive_online_local_wall_sec_sum": adaptive_online_metrics["local_wall_sec_sum"],
                "adaptive_online_local_wall_sec_count": adaptive_online_metrics[
                    "local_wall_sec_count"
                ],
                "adaptive_online_local_llm_call_count_sum": adaptive_online_metrics[
                    "local_llm_call_count_sum"
                ],
                "adaptive_online_local_llm_call_count_count": adaptive_online_metrics[
                    "local_llm_call_count_count"
                ],
                "adaptive_online_reward_progress_score_sum": adaptive_online_metrics[
                    "reward_progress_score_sum"
                ],
                "adaptive_online_reward_progress_score_count": adaptive_online_metrics[
                    "reward_progress_score_count"
                ],
                "adaptive_online_progress_reason_counts": adaptive_online_metrics[
                    "progress_reason_counts"
                ],
                "adaptive_online_delayed_rescue_bonus_sum": adaptive_online_metrics[
                    "delayed_rescue_bonus_sum"
                ],
                "adaptive_online_delayed_rescue_bonus_count": adaptive_online_metrics[
                    "delayed_rescue_bonus_count"
                ],
                "adaptive_online_credited_rescues_plus_1": adaptive_online_metrics[
                    "credited_rescues_plus_1"
                ],
                "adaptive_online_credited_rescues_plus_2": adaptive_online_metrics[
                    "credited_rescues_plus_2"
                ],
                "adaptive_budget_policy_mode": adaptive_budget_policy_metrics[
                    "adaptive_budget_policy_mode"
                ],
                "adaptive_budget_policy_algo": adaptive_budget_policy_metrics[
                    "adaptive_budget_policy_algo"
                ],
                "adaptive_budget_policy_architecture": adaptive_budget_policy_metrics[
                    "adaptive_budget_policy_architecture"
                ],
                "adaptive_budget_decisions": adaptive_budget_policy_metrics["decisions"],
                "adaptive_budget_selected_increase_attempts": adaptive_budget_policy_metrics[
                    "selected_increase_attempts"
                ],
                "adaptive_budget_applied_increase_attempts": adaptive_budget_policy_metrics[
                    "applied_increase_attempts"
                ],
                "adaptive_budget_selected_time_attempts": adaptive_budget_policy_metrics[
                    "selected_time_attempts"
                ],
                "adaptive_budget_selected_tokens_attempts": adaptive_budget_policy_metrics[
                    "selected_tokens_attempts"
                ],
                "adaptive_budget_selected_context_attempts": adaptive_budget_policy_metrics[
                    "selected_context_attempts"
                ],
                "adaptive_budget_applied_time_attempts": adaptive_budget_policy_metrics[
                    "applied_time_attempts"
                ],
                "adaptive_budget_applied_tokens_attempts": adaptive_budget_policy_metrics[
                    "applied_tokens_attempts"
                ],
                "adaptive_budget_applied_context_attempts": adaptive_budget_policy_metrics[
                    "applied_context_attempts"
                ],
                "adaptive_budget_override_attempts": adaptive_budget_policy_metrics[
                    "override_attempts"
                ],
                "adaptive_budget_updates": adaptive_budget_policy_metrics["updates"],
                "adaptive_budget_reward_sum": adaptive_budget_policy_metrics["reward_sum"],
                "adaptive_budget_reward_count": adaptive_budget_policy_metrics["reward_count"],
                "adaptive_budget_reward_values": adaptive_budget_policy_metrics["reward_values"],
                "adaptive_budget_propensity_sum": adaptive_budget_policy_metrics["propensity_sum"],
                "adaptive_budget_propensity_count": adaptive_budget_policy_metrics[
                    "propensity_count"
                ],
                "adaptive_budget_family_propensity_sum": adaptive_budget_policy_metrics[
                    "family_propensity_sum"
                ],
                "adaptive_budget_family_propensity_count": adaptive_budget_policy_metrics[
                    "family_propensity_count"
                ],
                "adaptive_budget_request_budget_exceeded_attempts": adaptive_budget_policy_metrics[
                    "request_budget_exceeded_attempts"
                ],
                "adaptive_budget_applied_multiplier_sum": adaptive_budget_policy_metrics[
                    "applied_multiplier_sum"
                ],
                "adaptive_budget_applied_multiplier_count": adaptive_budget_policy_metrics[
                    "applied_multiplier_count"
                ],
                "adaptive_budget_local_tokens_total_sum": adaptive_budget_policy_metrics[
                    "local_tokens_total_sum"
                ],
                "adaptive_budget_local_tokens_total_count": adaptive_budget_policy_metrics[
                    "local_tokens_total_count"
                ],
                "adaptive_budget_local_wall_sec_sum": adaptive_budget_policy_metrics[
                    "local_wall_sec_sum"
                ],
                "adaptive_budget_local_wall_sec_count": adaptive_budget_policy_metrics[
                    "local_wall_sec_count"
                ],
                "adaptive_budget_local_llm_call_count_sum": adaptive_budget_policy_metrics[
                    "local_llm_call_count_sum"
                ],
                "adaptive_budget_local_llm_call_count_count": adaptive_budget_policy_metrics[
                    "local_llm_call_count_count"
                ],
                "adaptive_budget_reward_progress_score_sum": adaptive_budget_policy_metrics[
                    "reward_progress_score_sum"
                ],
                "adaptive_budget_reward_progress_score_count": adaptive_budget_policy_metrics[
                    "reward_progress_score_count"
                ],
                "adaptive_budget_propose_budget_change_attempts": adaptive_budget_policy_metrics[
                    "propose_budget_change_attempts"
                ],
                "adaptive_budget_max_tokens_change_attempts": adaptive_budget_policy_metrics[
                    "max_tokens_change_attempts"
                ],
                "adaptive_budget_helper_allowance_change_attempts": adaptive_budget_policy_metrics[
                    "helper_allowance_change_attempts"
                ],
                "adaptive_budget_stage_a_requests_change_attempts": (
                    adaptive_budget_policy_metrics["stage_a_requests_change_attempts"]
                ),
                "adaptive_budget_stage_a_chars_change_attempts": adaptive_budget_policy_metrics[
                    "stage_a_chars_change_attempts"
                ],
                "adaptive_budget_retry_history_budget_change_attempts": (
                    adaptive_budget_policy_metrics["retry_history_budget_change_attempts"]
                ),
                "adaptive_budget_compaction_allowance_change_attempts": (
                    adaptive_budget_policy_metrics["compaction_allowance_change_attempts"]
                ),
                "adaptive_budget_response_mode_change_attempts": adaptive_budget_policy_metrics[
                    "response_mode_change_attempts"
                ],
                "adaptive_budget_partial_override_attempts": adaptive_budget_policy_metrics[
                    "partial_override_attempts"
                ],
                "adaptive_budget_effective_helper_max_calls_sum": adaptive_budget_policy_metrics[
                    "effective_helper_max_calls_sum"
                ],
                "adaptive_budget_effective_helper_max_calls_count": adaptive_budget_policy_metrics[
                    "effective_helper_max_calls_count"
                ],
                "adaptive_budget_effective_stage_a_max_requests_sum": (
                    adaptive_budget_policy_metrics["effective_stage_a_max_requests_sum"]
                ),
                "adaptive_budget_effective_stage_a_max_requests_count": (
                    adaptive_budget_policy_metrics["effective_stage_a_max_requests_count"]
                ),
                "adaptive_budget_effective_stage_a_max_chars_sum": adaptive_budget_policy_metrics[
                    "effective_stage_a_max_chars_sum"
                ],
                "adaptive_budget_effective_stage_a_max_chars_count": (
                    adaptive_budget_policy_metrics["effective_stage_a_max_chars_count"]
                ),
                "adaptive_budget_effective_retry_history_max_chars_sum": (
                    adaptive_budget_policy_metrics["effective_retry_history_max_chars_sum"]
                ),
                "adaptive_budget_effective_retry_history_max_chars_count": (
                    adaptive_budget_policy_metrics["effective_retry_history_max_chars_count"]
                ),
                "adaptive_budget_effective_context_compaction_max_calls_sum": (
                    adaptive_budget_policy_metrics["effective_context_compaction_max_calls_sum"]
                ),
                "adaptive_budget_effective_context_compaction_max_calls_count": (
                    adaptive_budget_policy_metrics["effective_context_compaction_max_calls_count"]
                ),
                "adaptive_budget_effective_openai_max_tokens_sum": (
                    adaptive_budget_policy_metrics["effective_openai_max_tokens_sum"]
                ),
                "adaptive_budget_effective_openai_max_tokens_count": (
                    adaptive_budget_policy_metrics["effective_openai_max_tokens_count"]
                ),
                "syntax_preflight_enabled_attempts": syntax_preflight_metrics["enabled_attempts"],
                "syntax_preflight_cycles_sum": syntax_preflight_metrics["cycles_sum"],
                "syntax_preflight_failures_sum": syntax_preflight_metrics["failures_sum"],
                "syntax_preflight_failed_attempts": syntax_preflight_metrics["failed_attempts"],
                "patch_verifier_attempts": patch_verifier_scope_metrics["verifier_attempts"],
                "patch_verifier_source_scope_matches": patch_verifier_scope_metrics[
                    "source_scope_matches"
                ],
                "patch_verifier_recovery_eligible_attempts": patch_verifier_scope_metrics[
                    "recovery_eligible_attempts"
                ],
                "patch_verifier_recovery_invoked_attempts": patch_verifier_scope_metrics[
                    "recovery_invoked_attempts"
                ],
                "patch_verifier_recovery_passed_attempts": patch_verifier_scope_metrics[
                    "recovery_passed_attempts"
                ],
                "patch_verifier_recovery_result_counts": patch_verifier_scope_metrics[
                    "recovery_result_counts"
                ],
                "patch_verifier_recovery_terminal_note_counts": patch_verifier_scope_metrics[
                    "recovery_terminal_note_counts"
                ],
                "retry_history_attempts": retry_history_stats["attempts"],
                "retry_history_chars_sum": retry_history_stats["chars_sum"],
                "retry_history_turns_sum": retry_history_stats["turns_sum"],
                "retry_history_budget_applied_attempts": retry_history_stats[
                    "budget_applied_attempts"
                ],
                "retry_history_truncated_attempts": retry_history_stats["truncated_attempts"],
            }
        )
    return rows


def _records_by_case_condition(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Index deduped records by (run_id, case_id, condition) key."""
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (run_id, case_id, condition), records in grouped.items():
        out[(str(run_id), str(case_id), str(condition))] = _dedupe_records_by_attempt(records)
    return out


def _summarize_bucket(
    rows: list[dict[str, Any]],
    *,
    records_by_case_condition: dict[tuple[str, str, str], list[dict[str, Any]]] | None = None,
    adaptive_ope_min_action_support: int = ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
    adaptive_ope_propensity_floor: float = ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
) -> dict[str, Any]:
    """Compute per-condition success rates and aggregate metrics for a bucket.

    Produces a dict keyed by condition name, each containing success
    rates, token usage, runtime, helper, online-policy, budget-policy,
    OPE summaries, and pairwise uplift comparisons.
    """
    min_action_support = max(1, int(adaptive_ope_min_action_support))
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    out: dict[str, Any] = {}
    for condition, all_condition_rows in sorted(by_condition.items()):
        # BAS-01 / D3: baselines are scored on the cases they can represent; the excluded
        # count is reported so the comparison's N stays explicit.
        condition_rows = [row for row in all_condition_rows if row.get("baseline_supported", True)]
        baseline_unsupported_cases = len(all_condition_rows) - len(condition_rows)
        successes = sum(1 for row in condition_rows if bool(row["success"]))
        cases = len(condition_rows)
        success_rate = (successes / cases) if cases else 0.0
        successful_attempts = [
            int(row["attempts_to_success"])
            for row in condition_rows
            if row["attempts_to_success"] is not None and bool(row["success"])
        ]
        token_values = [
            float(row["token_usage"]) for row in condition_rows if row["token_usage"] is not None
        ]
        token_prompt_values = [
            float(row.get("token_usage_prompt"))
            for row in condition_rows
            if row.get("token_usage_prompt") is not None
        ]
        token_completion_values = [
            float(row.get("token_usage_completion"))
            for row in condition_rows
            if row.get("token_usage_completion") is not None
        ]
        token_total_values = [
            float(row.get("token_usage_total"))
            for row in condition_rows
            if row.get("token_usage_total") is not None
        ]
        token_reasoning_values = [
            float(row.get("token_usage_reasoning"))
            for row in condition_rows
            if bool(row.get("reasoning_token_usage_available"))
            and row.get("token_usage_reasoning") is not None
        ]
        token_per_attempt_values = [
            float(row.get("token_usage_per_attempt_mean"))
            for row in condition_rows
            if row.get("token_usage_per_attempt_mean") is not None
        ]
        reasoning_token_per_attempt_values = [
            float(row.get("reasoning_token_usage_per_attempt_mean"))
            for row in condition_rows
            if bool(row.get("reasoning_token_usage_available"))
            and row.get("reasoning_token_usage_per_attempt_mean") is not None
        ]
        runtime_values = [
            float(row["runtime_sec"]) for row in condition_rows if row["runtime_sec"] is not None
        ]
        attempts_budget_values = [
            float(row["attempts_budget"])
            for row in condition_rows
            if isinstance(row.get("attempts_budget"), (int, float))
        ]
        failed_attempt_values = [
            float(row["attempts_executed"])
            for row in condition_rows
            if (not bool(row.get("success")))
            and isinstance(row.get("attempts_executed"), (int, float))
        ]
        successful_attempt_values = [
            float(row["attempts_to_success"])
            for row in condition_rows
            if bool(row.get("success")) and isinstance(row.get("attempts_to_success"), (int, float))
        ]
        success_attempt_ratio_values = [
            float(row.get("success_attempt_ratio"))
            for row in condition_rows
            if row.get("success_attempt_ratio") is not None
        ]
        llm_call_values = [
            float(row["llm_calls_total"])
            for row in condition_rows
            if isinstance(row.get("llm_calls_total"), (int, float))
        ]
        is_external_baseline = _is_external_baseline_condition(condition)
        baseline_teacher_duration_values = (
            [
                float(row["baseline_teacher_duration_sec_total"])
                for row in condition_rows
                if isinstance(row.get("baseline_teacher_duration_sec_total"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_learner_duration_values = (
            [
                float(row["baseline_learner_duration_sec_total"])
                for row in condition_rows
                if isinstance(row.get("baseline_learner_duration_sec_total"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_postcheck_duration_values = (
            [
                float(row["baseline_postcheck_duration_sec_total"])
                for row in condition_rows
                if isinstance(row.get("baseline_postcheck_duration_sec_total"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_learner_empty_response_counts = (
            [
                float(row["baseline_learner_empty_response_count"])
                for row in condition_rows
                if isinstance(row.get("baseline_learner_empty_response_count"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_attempts_with_postcheck_counts = (
            [
                float(row["baseline_attempts_with_postcheck_count"])
                for row in condition_rows
                if isinstance(row.get("baseline_attempts_with_postcheck_count"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_direct_gen_passed_values = (
            [
                1.0 if bool(row.get("baseline_direct_gen_passed")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_direct_gen_passed"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_debug_added_value_values = (
            [
                1.0 if bool(row.get("baseline_debug_added_value")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_debug_added_value"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_method_instrumentation_values = (
            [
                1.0 if bool(row.get("baseline_method_instrumentation")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_method_instrumentation"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_method_two_step_values = (
            [
                1.0 if bool(row.get("baseline_method_two_step_repair")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_method_two_step_repair"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_is_passing_values = (
            [
                1.0 if bool(row.get("baseline_is_passing")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_is_passing"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_feedback_improvement_values = (
            [
                float(row["baseline_test_feedback_improvement_steps"])
                for row in condition_rows
                if isinstance(row.get("baseline_test_feedback_improvement_steps"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_feedback_regression_values = (
            [
                float(row["baseline_test_feedback_regression_steps"])
                for row in condition_rows
                if isinstance(row.get("baseline_test_feedback_regression_steps"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_feedback_final_fail_values = (
            [
                float(row["baseline_test_feedback_final_fail_count"])
                for row in condition_rows
                if isinstance(row.get("baseline_test_feedback_final_fail_count"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_terminal_timeout_values = (
            [
                1.0 if bool(row.get("baseline_terminal_timeout")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_terminal_timeout"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_attempt_summary_count_total = (
            sum(int(row.get("baseline_attempt_summary_count") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_iteration_budget_exhausted_attempts_total = (
            sum(
                int(row.get("baseline_iteration_budget_exhausted_attempt_count") or 0)
                for row in condition_rows
            )
            if is_external_baseline
            else 0
        )
        baseline_no_write_buggy_py_attempts_total = (
            sum(
                int(row.get("baseline_no_write_buggy_py_attempt_count") or 0)
                for row in condition_rows
            )
            if is_external_baseline
            else 0
        )
        baseline_no_write_buggy_py_attempts_covered = (
            sum(
                int(row.get("baseline_no_write_buggy_py_attempt_covered") or 0)
                for row in condition_rows
            )
            if is_external_baseline
            else 0
        )
        baseline_no_test_attempts_total = (
            sum(int(row.get("baseline_no_test_attempt_count") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_no_test_attempts_covered = (
            sum(int(row.get("baseline_no_test_attempt_covered") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_subcall_count_total = (
            sum(int(row.get("baseline_subcall_count_total") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_subcall_count_attempts_covered = (
            sum(int(row.get("baseline_subcall_count_covered") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_test_run_count_total = (
            sum(int(row.get("baseline_test_run_count_total") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_test_run_count_attempts_covered = (
            sum(int(row.get("baseline_test_run_count_covered") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        stopped_reason_counts: dict[str, int] = {}
        fidelity_counts: dict[str, int] = {}
        repair_scope_counts: dict[str, int] = {}
        prompt_contract_counts: dict[str, int] = {}
        hybrid_guidance_mode_counts: dict[str, int] = {}
        hybrid_guidance_section_counts: dict[str, int] = {}
        output_contract_counts: dict[str, int] = {}
        upstream_commit_counts: dict[str, int] = {}
        contract_version_counts: dict[str, int] = {}
        hybrid_guidance_chars_values: list[float] = []
        if is_external_baseline:
            for row in condition_rows:
                value = row.get("baseline_stopped_reason")
                if isinstance(value, str) and value:
                    stopped_reason_counts[value] = int(stopped_reason_counts.get(value, 0) + 1)
                _increment_string_count(fidelity_counts, row.get("baseline_fidelity_level"))
                _increment_string_count(
                    repair_scope_counts, row.get("baseline_repair_scope_policy")
                )
                _increment_string_count(
                    prompt_contract_counts, row.get("baseline_prompt_contract_kind")
                )
                _increment_string_count(
                    hybrid_guidance_mode_counts, row.get("baseline_hybrid_guidance_mode")
                )
                guidance_chars = row.get("baseline_hybrid_guidance_chars")
                if isinstance(guidance_chars, (int, float)):
                    hybrid_guidance_chars_values.append(float(guidance_chars))
                sections_value = row.get("baseline_hybrid_guidance_sections")
                if isinstance(sections_value, list):
                    for section_name in sections_value:
                        _increment_string_count(hybrid_guidance_section_counts, section_name)
                _increment_string_count(output_contract_counts, row.get("baseline_output_contract"))
                _increment_string_count(upstream_commit_counts, row.get("baseline_upstream_commit"))
                _increment_string_count(
                    contract_version_counts, row.get("baseline_contract_version")
                )
        baseline_direct_gen_pass_rate_covered_cases = len(baseline_direct_gen_passed_values)
        baseline_debug_added_value_rate_covered_cases = len(baseline_debug_added_value_values)
        baseline_method_instrumentation_rate_covered_cases = len(
            baseline_method_instrumentation_values
        )
        baseline_method_two_step_repair_rate_covered_cases = len(baseline_method_two_step_values)
        baseline_is_passing_rate_covered_cases = len(baseline_is_passing_values)
        baseline_terminal_timeout_rate_covered_cases = len(baseline_terminal_timeout_values)
        baseline_feedback_improved_case_rate_covered_cases = len(
            baseline_feedback_improvement_values
        )
        baseline_feedback_regressed_case_rate_covered_cases = len(
            baseline_feedback_regression_values
        )
        total_patch_attempts = sum(int(row.get("patch_attempts") or 0) for row in condition_rows)
        total_test_touch_attempts = sum(
            int(row.get("test_touch_attempts") or 0) for row in condition_rows
        )
        solve_distribution: dict[str, int] = {}
        for row in condition_rows:
            if not bool(row.get("success")):
                continue
            value = row.get("attempts_to_success")
            if value is None:
                continue
            key = str(int(value))
            solve_distribution[key] = int(solve_distribution.get(key, 0) + 1)
        out[condition] = {
            "cases": cases,
            "successes": successes,
            "success_rate": success_rate,
            "baseline_unsupported_cases": baseline_unsupported_cases,
            "median_attempts_to_success": _median(successful_attempts),
            "mean_attempts_to_success": _mean(successful_attempt_values),
            "mean_attempts_when_failed": _mean(failed_attempt_values),
            "mean_attempts_budget": _mean(attempts_budget_values),
            "mean_attempt_ratio_when_successful": _mean(success_attempt_ratio_values),
            "solve_on_attempt_distribution": solve_distribution,
            "mean_token_usage": _mean(token_values),
            "mean_tokens_prompt": _mean(token_prompt_values),
            "mean_tokens_completion": _mean(token_completion_values),
            "mean_tokens_total": _mean(token_total_values),
            "mean_tokens_reasoning": _mean(token_reasoning_values),
            "mean_tokens_per_attempt": _mean(token_per_attempt_values),
            "mean_reasoning_tokens_per_attempt": _mean(reasoning_token_per_attempt_values),
            "mean_llm_calls_per_case": _mean(llm_call_values),
            "token_covered_cases": sum(
                1 for row in condition_rows if row["token_usage"] is not None
            ),
            "token_covered_attempts": sum(
                int(row.get("token_covered_attempts") or 0) for row in condition_rows
            ),
            "token_covered_fraction": _ratio(
                sum(int(row.get("token_covered_attempts") or 0) for row in condition_rows),
                sum(int(row.get("attempts_executed") or 0) for row in condition_rows),
            ),
            "reasoning_token_covered_cases": sum(
                1 for row in condition_rows if bool(row.get("reasoning_token_usage_available"))
            ),
            "reasoning_token_coverage_rate": _ratio(
                sum(
                    1 for row in condition_rows if bool(row.get("reasoning_token_usage_available"))
                ),
                cases,
            ),
            "mean_runtime_sec": _mean(runtime_values),
            "mean_timing_breakdown": _condition_timing_breakdown_mean(condition_rows),
            "mean_token_breakdown": _condition_token_breakdown_mean(condition_rows),
            "baseline_teacher_duration_sec_total_mean": _mean(baseline_teacher_duration_values),
            "baseline_learner_duration_sec_total_mean": _mean(baseline_learner_duration_values),
            "baseline_postcheck_duration_sec_total_mean": _mean(baseline_postcheck_duration_values),
            "baseline_learner_empty_response_count_mean": _mean(
                baseline_learner_empty_response_counts
            ),
            "baseline_attempts_with_postcheck_count_mean": _mean(
                baseline_attempts_with_postcheck_counts
            ),
            "baseline_direct_gen_pass_rate": _mean(baseline_direct_gen_passed_values),
            "baseline_direct_gen_pass_rate_covered_cases": baseline_direct_gen_pass_rate_covered_cases,
            "baseline_direct_gen_pass_rate_coverage_rate": _ratio(
                baseline_direct_gen_pass_rate_covered_cases, cases
            ),
            "baseline_debug_added_value_rate": _mean(baseline_debug_added_value_values),
            "baseline_debug_added_value_rate_covered_cases": (
                baseline_debug_added_value_rate_covered_cases
            ),
            "baseline_debug_added_value_rate_coverage_rate": _ratio(
                baseline_debug_added_value_rate_covered_cases, cases
            ),
            "baseline_method_instrumentation_rate": _mean(baseline_method_instrumentation_values),
            "baseline_method_instrumentation_rate_covered_cases": (
                baseline_method_instrumentation_rate_covered_cases
            ),
            "baseline_method_instrumentation_rate_coverage_rate": _ratio(
                baseline_method_instrumentation_rate_covered_cases, cases
            ),
            "baseline_method_two_step_repair_rate": _mean(baseline_method_two_step_values),
            "baseline_method_two_step_repair_rate_covered_cases": (
                baseline_method_two_step_repair_rate_covered_cases
            ),
            "baseline_method_two_step_repair_rate_coverage_rate": _ratio(
                baseline_method_two_step_repair_rate_covered_cases, cases
            ),
            "baseline_is_passing_rate": _mean(baseline_is_passing_values),
            "baseline_is_passing_rate_covered_cases": baseline_is_passing_rate_covered_cases,
            "baseline_is_passing_rate_coverage_rate": _ratio(
                baseline_is_passing_rate_covered_cases, cases
            ),
            "baseline_terminal_timeout_rate": _mean(baseline_terminal_timeout_values),
            "baseline_terminal_timeout_rate_covered_cases": (
                baseline_terminal_timeout_rate_covered_cases
            ),
            "baseline_terminal_timeout_rate_coverage_rate": _ratio(
                baseline_terminal_timeout_rate_covered_cases, cases
            ),
            "baseline_feedback_improvement_steps_mean": _mean(baseline_feedback_improvement_values),
            "baseline_feedback_regression_steps_mean": _mean(baseline_feedback_regression_values),
            "baseline_feedback_improved_case_rate": _mean(
                [1.0 if value > 0.0 else 0.0 for value in baseline_feedback_improvement_values]
            ),
            "baseline_feedback_improved_case_rate_covered_cases": (
                baseline_feedback_improved_case_rate_covered_cases
            ),
            "baseline_feedback_improved_case_rate_coverage_rate": _ratio(
                baseline_feedback_improved_case_rate_covered_cases, cases
            ),
            "baseline_feedback_regressed_case_rate": _mean(
                [1.0 if value > 0.0 else 0.0 for value in baseline_feedback_regression_values]
            ),
            "baseline_feedback_regressed_case_rate_covered_cases": (
                baseline_feedback_regressed_case_rate_covered_cases
            ),
            "baseline_feedback_regressed_case_rate_coverage_rate": _ratio(
                baseline_feedback_regressed_case_rate_covered_cases, cases
            ),
            "baseline_feedback_final_fail_count_mean": _mean(baseline_feedback_final_fail_values),
            "baseline_attempt_summary_count_total": baseline_attempt_summary_count_total,
            "baseline_iteration_budget_exhausted_attempts_total": (
                baseline_iteration_budget_exhausted_attempts_total
            ),
            "baseline_iteration_budget_exhausted_attempt_rate": _ratio(
                baseline_iteration_budget_exhausted_attempts_total,
                baseline_attempt_summary_count_total,
            ),
            "baseline_no_write_buggy_py_attempts_total": baseline_no_write_buggy_py_attempts_total,
            "baseline_no_write_buggy_py_attempts_covered": (
                baseline_no_write_buggy_py_attempts_covered
            ),
            "baseline_no_write_buggy_py_attempt_rate": _ratio(
                baseline_no_write_buggy_py_attempts_total,
                baseline_no_write_buggy_py_attempts_covered,
            ),
            "baseline_no_test_attempts_total": baseline_no_test_attempts_total,
            "baseline_no_test_attempts_covered": baseline_no_test_attempts_covered,
            "baseline_no_test_attempt_rate": _ratio(
                baseline_no_test_attempts_total,
                baseline_no_test_attempts_covered,
            ),
            "baseline_subcall_count_total": baseline_subcall_count_total,
            "baseline_subcall_count_attempts_covered": baseline_subcall_count_attempts_covered,
            "baseline_subcall_count_per_attempt": (
                baseline_subcall_count_total / baseline_subcall_count_attempts_covered
                if baseline_subcall_count_attempts_covered > 0
                else None
            ),
            "baseline_test_run_count_total": baseline_test_run_count_total,
            "baseline_test_run_count_attempts_covered": (baseline_test_run_count_attempts_covered),
            "baseline_test_run_count_per_attempt": (
                baseline_test_run_count_total / baseline_test_run_count_attempts_covered
                if baseline_test_run_count_attempts_covered > 0
                else None
            ),
            "baseline_stopped_reason_counts": stopped_reason_counts,
            "baseline_fidelity_level_counts": fidelity_counts,
            "baseline_repair_scope_policy_counts": repair_scope_counts,
            "baseline_prompt_contract_kind_counts": prompt_contract_counts,
            "baseline_hybrid_guidance_mode_counts": hybrid_guidance_mode_counts,
            "baseline_hybrid_guidance_section_counts": hybrid_guidance_section_counts,
            "baseline_hybrid_guidance_chars_mean": _mean(hybrid_guidance_chars_values),
            "baseline_output_contract_counts": output_contract_counts,
            "baseline_upstream_commit_counts": upstream_commit_counts,
            "baseline_contract_version_counts": contract_version_counts,
            "patch_attempts": total_patch_attempts,
            "test_touch_attempts": total_test_touch_attempts,
            "test_touch_attempt_rate": (
                (total_test_touch_attempts / total_patch_attempts)
                if total_patch_attempts > 0
                else None
            ),
            "relaxed_apply_attempts": sum(
                int(row.get("relaxed_apply_attempts") or 0) for row in condition_rows
            ),
            "test_surface_modified_attempts": sum(
                int(row.get("test_surface_modified_attempts") or 0) for row in condition_rows
            ),
            **_adaptive_rollup_metrics(
                condition_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
                include_condition_only_counts=True,
            ),
        }
    out["condition_matrix"] = [
        {"condition": condition, **dict(stats)}
        for condition, stats in sorted(
            (
                (condition, stats)
                for condition, stats in out.items()
                if isinstance(stats, dict) and "cases" in stats and "success_rate" in stats
            ),
            key=lambda item: item[0],
        )
    ]
    out["pairwise_uplifts"] = _pairwise_uplifts(summary=out)

    out["uplift_success_rate"] = _uplift_success_rate(out)
    stage_b_prompt_values = [
        float(row["stage_b_prompt_chars_mean"])
        for row in rows
        if row.get("stage_b_prompt_chars_mean") is not None
    ]
    out["mean_prompt_chars_stage_b"] = _mean(stage_b_prompt_values)
    out["mean_prompt_chars_stage_b_by_condition"] = _mean_prompt_chars_by_condition(rows)
    retry_history_attempts = sum(int(row.get("retry_history_attempts") or 0) for row in rows)
    retry_history_chars_sum = sum(float(row.get("retry_history_chars_sum") or 0.0) for row in rows)
    retry_history_turns_sum = sum(float(row.get("retry_history_turns_sum") or 0.0) for row in rows)
    retry_history_budget_applied_attempts = sum(
        int(row.get("retry_history_budget_applied_attempts") or 0) for row in rows
    )
    retry_history_truncated_attempts = sum(
        int(row.get("retry_history_truncated_attempts") or 0) for row in rows
    )
    out["mean_retry_history_chars"] = (
        (retry_history_chars_sum / retry_history_attempts) if retry_history_attempts > 0 else None
    )
    out["mean_retry_history_turns"] = (
        (retry_history_turns_sum / retry_history_attempts) if retry_history_attempts > 0 else None
    )
    out["retry_history_budget_applied_rate"] = (
        (retry_history_budget_applied_attempts / retry_history_attempts)
        if retry_history_attempts > 0
        else None
    )
    out["retry_history_truncation_rate"] = (
        (retry_history_truncated_attempts / retry_history_attempts)
        if retry_history_attempts > 0
        else None
    )

    stage_a_attempts = sum(int(row.get("stage_a_request_attempts") or 0) for row in rows)
    stage_a_successes = sum(int(row.get("stage_a_request_successes") or 0) for row in rows)
    out["stage_a_request_success_rate"] = (
        (stage_a_successes / stage_a_attempts) if stage_a_attempts > 0 else None
    )
    retry_gate_attempts = sum(int(row.get("retry_gate_attempts") or 0) for row in rows)
    retry_gate_blocks = sum(int(row.get("retry_gate_blocks") or 0) for row in rows)
    out["retry_gate_block_rate"] = (
        (retry_gate_blocks / retry_gate_attempts) if retry_gate_attempts > 0 else None
    )
    total_patch_attempts = sum(int(row.get("patch_attempts") or 0) for row in rows)
    total_test_touch_attempts = sum(int(row.get("test_touch_attempts") or 0) for row in rows)
    out["patch_attempts"] = total_patch_attempts
    out["test_touch_attempts"] = total_test_touch_attempts
    out["relaxed_apply_attempts"] = sum(int(row.get("relaxed_apply_attempts") or 0) for row in rows)
    out["test_surface_modified_attempts"] = sum(
        int(row.get("test_surface_modified_attempts") or 0) for row in rows
    )
    out["test_touch_attempt_rate"] = (
        (total_test_touch_attempts / total_patch_attempts) if total_patch_attempts > 0 else None
    )
    out.update(
        _adaptive_rollup_metrics(
            rows,
            records_by_case_condition=records_by_case_condition,
            adaptive_ope_min_action_support=min_action_support,
            adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            include_condition_only_counts=False,
        )
    )

    staged_rows = [
        row
        for row in rows
        if row.get("condition") == "layered_adaptive_llm"
        and row.get("context_protocol") == "staged"
    ]
    monolithic_rows = [
        row
        for row in rows
        if row.get("condition") == "layered_adaptive_llm"
        and row.get("context_protocol") == "monolithic"
    ]
    if staged_rows and monolithic_rows:
        staged_rate = sum(1 for row in staged_rows if bool(row.get("success"))) / len(staged_rows)
        monolithic_rate = sum(1 for row in monolithic_rows if bool(row.get("success"))) / len(
            monolithic_rows
        )
        out["uplift_staged_vs_monolithic"] = staged_rate - monolithic_rate
    else:
        out["uplift_staged_vs_monolithic"] = None
    out["uplift_layered_gated_vs_base"] = _uplift_between_conditions(
        summary=out,
        a_condition="base_repair",
        b_condition="layered_adaptive_gated",
    )
    out["uplift_layered_llm_vs_base"] = _uplift_between_conditions(
        summary=out,
        a_condition="base_repair",
        b_condition="layered_adaptive_llm",
    )
    return out


def _baseline_patch_meta_last_str(records: list[dict[str, Any]], key: str) -> str | None:
    for record in reversed(records):
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        value = patch_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _baseline_patch_meta_last_int(records: list[dict[str, Any]], key: str) -> int | None:
    for record in reversed(records):
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        value = patch_meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _baseline_patch_meta_last_str_list(records: list[dict[str, Any]], key: str) -> list[str]:
    for record in reversed(records):
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        value = patch_meta.get(key)
        if not isinstance(value, list):
            continue
        return [str(item) for item in value if str(item).strip()]
    return []


def _baseline_meta_from_record(record: dict[str, Any]) -> dict[str, Any]:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return {}
    baseline_meta = patch_meta.get("baseline_meta")
    return baseline_meta if isinstance(baseline_meta, dict) else {}


def _baseline_meta_last_str(records: list[dict[str, Any]], key: str) -> str | None:
    for record in reversed(records):
        value = _baseline_meta_from_record(record).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _baseline_attempt_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in reversed(records):
        attempts = _baseline_meta_from_record(record).get("attempts")
        if not isinstance(attempts, list):
            continue
        return [cast(dict[str, Any], attempt) for attempt in attempts if isinstance(attempt, dict)]
    return []


def _baseline_attempt_rollup(records: list[dict[str, Any]]) -> dict[str, int | None]:
    attempts = _baseline_attempt_summaries(records)
    attempt_count = len(attempts)
    if attempt_count <= 0:
        return {
            "attempt_count": 0,
            "iteration_budget_exhausted_attempt_count": None,
            "no_write_buggy_py_attempt_count": None,
            "no_write_buggy_py_attempt_covered": 0,
            "no_test_attempt_count": None,
            "no_test_attempt_covered": 0,
            "subcall_count_total": None,
            "subcall_count_covered": 0,
            "test_run_count_total": None,
            "test_run_count_covered": 0,
        }
    iteration_budget_exhausted_attempt_count = 0
    no_write_buggy_py_attempt_count = 0
    no_write_buggy_py_attempt_covered = 0
    no_test_attempt_count = 0
    subcall_count_total = 0
    subcall_count_covered = 0
    test_run_count_total = 0
    for attempt in attempts:
        if str(attempt.get("stop_reason") or "").strip() == "iteration_budget_exhausted":
            iteration_budget_exhausted_attempt_count += 1
        no_write = attempt.get("no_write_buggy_py")
        if isinstance(no_write, bool):
            no_write_buggy_py_attempt_covered += 1
            if no_write:
                no_write_buggy_py_attempt_count += 1
        else:
            write_calls = attempt.get("write_buggy_py_calls")
            if isinstance(write_calls, int) and not isinstance(write_calls, bool):
                no_write_buggy_py_attempt_covered += 1
                if write_calls <= 0:
                    no_write_buggy_py_attempt_count += 1
        subcall_count = attempt.get("subcall_count")
        if isinstance(subcall_count, int) and not isinstance(subcall_count, bool):
            subcall_count_total += max(0, subcall_count)
            subcall_count_covered += 1
        test_run_count = attempt.get("test_run_count")
        if isinstance(test_run_count, int) and not isinstance(test_run_count, bool):
            normalized_test_run_count = max(0, test_run_count)
            test_run_count_total += normalized_test_run_count
            if normalized_test_run_count <= 0:
                no_test_attempt_count += 1
            continue
        if isinstance(attempt.get("test_returncode"), int):
            test_run_count_total += 1
        else:
            no_test_attempt_count += 1
    return {
        "attempt_count": attempt_count,
        "iteration_budget_exhausted_attempt_count": iteration_budget_exhausted_attempt_count,
        "no_write_buggy_py_attempt_count": (
            no_write_buggy_py_attempt_count if no_write_buggy_py_attempt_covered > 0 else None
        ),
        "no_write_buggy_py_attempt_covered": no_write_buggy_py_attempt_covered,
        "no_test_attempt_count": no_test_attempt_count,
        "no_test_attempt_covered": attempt_count,
        "subcall_count_total": subcall_count_total if subcall_count_covered > 0 else None,
        "subcall_count_covered": subcall_count_covered,
        "test_run_count_total": test_run_count_total,
        "test_run_count_covered": attempt_count,
    }


def _increment_string_count(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, str):
        return
    normalized = value.strip()
    if not normalized:
        return
    target[normalized] = int(target.get(normalized, 0) + 1)


def _adaptive_rollup_metrics(
    rows: list[dict[str, Any]],
    *,
    records_by_case_condition: dict[tuple[str, str, str], list[dict[str, Any]]] | None,
    adaptive_ope_min_action_support: int,
    adaptive_ope_propensity_floor: float,
    include_condition_only_counts: bool,
) -> dict[str, Any]:
    """Aggregate adaptive helper, online-policy, and budget-policy metrics.

    Rolls up per-row attempt-level counters into bucket-wide totals,
    rates, and off-policy evaluation summaries.
    """
    total_attempts_executed = sum(int(row.get("attempts_executed") or 0) for row in rows)
    helper_invoked_cases = sum(1 for row in rows if bool(row.get("helper_invoked_case")))
    helper_materialized_cases = sum(1 for row in rows if bool(row.get("helper_materialized_case")))
    helper_invocation_attempts_total = sum(
        int(row.get("helper_invocation_attempts") or 0) for row in rows
    )
    helper_materialized_attempts_total = sum(
        int(row.get("helper_materialized_attempts") or 0) for row in rows
    )
    helper_low_yield_attempts_total = sum(
        int(row.get("helper_low_yield_attempts") or 0) for row in rows
    )
    helper_transport_failure_attempts_total = sum(
        int(row.get("helper_transport_failure_attempts") or 0) for row in rows
    )
    helper_backend_unavailable_attempts_total = sum(
        int(row.get("helper_backend_unavailable_attempts") or 0) for row in rows
    )
    helper_compat_reasoning_only_attempts_total = sum(
        int(row.get("helper_compat_reasoning_only_attempts") or 0) for row in rows
    )
    helper_invalid_payload_attempts_total = sum(
        int(row.get("helper_invalid_payload_attempts") or 0) for row in rows
    )
    helper_request_budget_exceeded_attempts_total = sum(
        int(row.get("helper_request_budget_exceeded_attempts") or 0) for row in rows
    )
    helper_retry_attempts_total = sum(int(row.get("helper_retry_attempts") or 0) for row in rows)
    helper_transport_retry_attempts_total = sum(
        int(row.get("helper_transport_retry_attempts") or 0) for row in rows
    )
    helper_effective_attempts_total = sum(
        int(row.get("helper_effective_attempts") or 0) for row in rows
    )
    helper_success_with_invocation_cases = sum(
        1 for row in rows if bool(row.get("helper_success_with_invocation"))
    )
    helper_rescued_cases = sum(1 for row in rows if bool(row.get("helper_rescued")))
    helper_calls_used_values = [
        float(row.get("helper_calls_used_final"))
        for row in rows
        if isinstance(row.get("helper_calls_used_final"), (int, float))
    ]
    helper_decision_attempts_total = sum(
        int(row.get("helper_decision_attempts") or 0) for row in rows
    )
    helper_need_more_true_attempts_total = sum(
        int(row.get("helper_need_more_true_attempts") or 0) for row in rows
    )
    helper_need_more_false_attempts_total = sum(
        int(row.get("helper_need_more_false_attempts") or 0) for row in rows
    )
    helper_need_more_labeled_attempts = (
        helper_need_more_true_attempts_total + helper_need_more_false_attempts_total
    )
    helper_budget_exhausted_attempts_total = sum(
        int(row.get("helper_budget_exhausted_attempts") or 0) for row in rows
    )
    helper_saturation_guard_attempts_total = sum(
        int(row.get("helper_saturation_guard_attempts") or 0) for row in rows
    )
    adaptive_budget_topup_attempts_total = sum(
        int(row.get("adaptive_budget_topup_attempts") or 0) for row in rows
    )
    adaptive_budget_topup_cases = sum(
        1 for row in rows if bool(row.get("adaptive_budget_topup_case"))
    )
    adaptive_online_decisions_total = sum(
        int(row.get("adaptive_online_decisions") or 0) for row in rows
    )
    adaptive_online_selected_action_counts_total: dict[str, int] = defaultdict(int)
    adaptive_online_applied_action_counts_total: dict[str, int] = defaultdict(int)
    adaptive_online_behavior_action_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        selected_counts = row.get("adaptive_online_selected_action_counts")
        if isinstance(selected_counts, dict):
            for action in ADAPTIVE_ONLINE_ACTIONS:
                adaptive_online_selected_action_counts_total[action] += int(
                    selected_counts.get(action) or 0
                )
        applied_counts = row.get("adaptive_online_applied_action_counts")
        if isinstance(applied_counts, dict):
            for action in ADAPTIVE_ONLINE_ACTIONS:
                adaptive_online_applied_action_counts_total[action] += int(
                    applied_counts.get(action) or 0
                )
        behavior_counts = row.get("adaptive_online_behavior_action_counts")
        if isinstance(behavior_counts, dict):
            for action in ADAPTIVE_ONLINE_ACTIONS:
                adaptive_online_behavior_action_counts_total[action] += int(
                    behavior_counts.get(action) or 0
                )
    adaptive_online_selected_invoke_total = sum(
        int(row.get("adaptive_online_selected_invoke_attempts") or 0) for row in rows
    )
    adaptive_online_applied_invoke_total = sum(
        int(row.get("adaptive_online_applied_invoke_attempts") or 0) for row in rows
    )
    adaptive_online_override_total = sum(
        int(row.get("adaptive_online_override_attempts") or 0) for row in rows
    )
    adaptive_online_updates_total = sum(
        int(row.get("adaptive_online_updates") or 0) for row in rows
    )
    adaptive_online_reward_sum_total = sum(
        float(row.get("adaptive_online_reward_sum") or 0.0) for row in rows
    )
    adaptive_online_reward_count_total = sum(
        int(row.get("adaptive_online_reward_count") or 0) for row in rows
    )
    adaptive_online_reward_values_total: list[float] = []
    for row in rows:
        reward_values = row.get("adaptive_online_reward_values")
        if not isinstance(reward_values, list):
            continue
        for value in reward_values:
            parsed = _as_float_or_none(value)
            if parsed is not None:
                adaptive_online_reward_values_total.append(parsed)
    adaptive_online_propensity_sum_total = sum(
        float(row.get("adaptive_online_propensity_sum") or 0.0) for row in rows
    )
    adaptive_online_propensity_count_total = sum(
        int(row.get("adaptive_online_propensity_count") or 0) for row in rows
    )
    adaptive_online_local_tokens_total_sum_total = sum(
        float(row.get("adaptive_online_local_tokens_total_sum") or 0.0) for row in rows
    )
    adaptive_online_local_tokens_total_count_total = sum(
        int(row.get("adaptive_online_local_tokens_total_count") or 0) for row in rows
    )
    adaptive_online_local_wall_sec_sum_total = sum(
        float(row.get("adaptive_online_local_wall_sec_sum") or 0.0) for row in rows
    )
    adaptive_online_local_wall_sec_count_total = sum(
        int(row.get("adaptive_online_local_wall_sec_count") or 0) for row in rows
    )
    adaptive_online_local_llm_call_count_sum_total = sum(
        float(row.get("adaptive_online_local_llm_call_count_sum") or 0.0) for row in rows
    )
    adaptive_online_local_llm_call_count_count_total = sum(
        int(row.get("adaptive_online_local_llm_call_count_count") or 0) for row in rows
    )
    adaptive_online_reward_progress_score_sum_total = sum(
        float(row.get("adaptive_online_reward_progress_score_sum") or 0.0) for row in rows
    )
    adaptive_online_reward_progress_score_count_total = sum(
        int(row.get("adaptive_online_reward_progress_score_count") or 0) for row in rows
    )
    adaptive_online_progress_reason_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        reason_counts = row.get("adaptive_online_progress_reason_counts")
        if not isinstance(reason_counts, dict):
            continue
        for reason, value in reason_counts.items():
            if isinstance(reason, str) and isinstance(value, (int, float)):
                adaptive_online_progress_reason_counts_total[reason] += int(value)
    adaptive_online_delayed_rescue_bonus_sum_total = sum(
        float(row.get("adaptive_online_delayed_rescue_bonus_sum") or 0.0) for row in rows
    )
    adaptive_online_delayed_rescue_bonus_count_total = sum(
        int(row.get("adaptive_online_delayed_rescue_bonus_count") or 0) for row in rows
    )
    adaptive_online_credited_rescues_plus_1_total = sum(
        int(row.get("adaptive_online_credited_rescues_plus_1") or 0) for row in rows
    )
    adaptive_online_credited_rescues_plus_2_total = sum(
        int(row.get("adaptive_online_credited_rescues_plus_2") or 0) for row in rows
    )
    adaptive_budget_decisions_total = sum(
        int(row.get("adaptive_budget_decisions") or 0) for row in rows
    )
    adaptive_budget_selected_increase_total = sum(
        int(row.get("adaptive_budget_selected_increase_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_increase_total = sum(
        int(row.get("adaptive_budget_applied_increase_attempts") or 0) for row in rows
    )
    adaptive_budget_selected_time_total = sum(
        int(row.get("adaptive_budget_selected_time_attempts") or 0) for row in rows
    )
    adaptive_budget_selected_tokens_total = sum(
        int(row.get("adaptive_budget_selected_tokens_attempts") or 0) for row in rows
    )
    adaptive_budget_selected_context_total = sum(
        int(row.get("adaptive_budget_selected_context_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_time_total = sum(
        int(row.get("adaptive_budget_applied_time_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_tokens_total = sum(
        int(row.get("adaptive_budget_applied_tokens_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_context_total = sum(
        int(row.get("adaptive_budget_applied_context_attempts") or 0) for row in rows
    )
    adaptive_budget_override_total = sum(
        int(row.get("adaptive_budget_override_attempts") or 0) for row in rows
    )
    adaptive_budget_updates_total = sum(
        int(row.get("adaptive_budget_updates") or 0) for row in rows
    )
    adaptive_budget_reward_sum_total = sum(
        float(row.get("adaptive_budget_reward_sum") or 0.0) for row in rows
    )
    adaptive_budget_reward_count_total = sum(
        int(row.get("adaptive_budget_reward_count") or 0) for row in rows
    )
    adaptive_budget_reward_values_total: list[float] = []
    for row in rows:
        reward_values = row.get("adaptive_budget_reward_values")
        if not isinstance(reward_values, list):
            continue
        for value in reward_values:
            parsed = _as_float_or_none(value)
            if parsed is not None:
                adaptive_budget_reward_values_total.append(parsed)
    adaptive_budget_propensity_sum_total = sum(
        float(row.get("adaptive_budget_propensity_sum") or 0.0) for row in rows
    )
    adaptive_budget_propensity_count_total = sum(
        int(row.get("adaptive_budget_propensity_count") or 0) for row in rows
    )
    adaptive_budget_family_propensity_sum_total = sum(
        float(row.get("adaptive_budget_family_propensity_sum") or 0.0) for row in rows
    )
    adaptive_budget_family_propensity_count_total = sum(
        int(row.get("adaptive_budget_family_propensity_count") or 0) for row in rows
    )
    adaptive_budget_request_budget_exceeded_total = sum(
        int(row.get("adaptive_budget_request_budget_exceeded_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_multiplier_sum_total = sum(
        float(row.get("adaptive_budget_applied_multiplier_sum") or 0.0) for row in rows
    )
    adaptive_budget_applied_multiplier_count_total = sum(
        int(row.get("adaptive_budget_applied_multiplier_count") or 0) for row in rows
    )
    adaptive_budget_local_tokens_total_sum_total = sum(
        float(row.get("adaptive_budget_local_tokens_total_sum") or 0.0) for row in rows
    )
    adaptive_budget_local_tokens_total_count_total = sum(
        int(row.get("adaptive_budget_local_tokens_total_count") or 0) for row in rows
    )
    adaptive_budget_local_wall_sec_sum_total = sum(
        float(row.get("adaptive_budget_local_wall_sec_sum") or 0.0) for row in rows
    )
    adaptive_budget_local_wall_sec_count_total = sum(
        int(row.get("adaptive_budget_local_wall_sec_count") or 0) for row in rows
    )
    adaptive_budget_local_llm_call_count_sum_total = sum(
        float(row.get("adaptive_budget_local_llm_call_count_sum") or 0.0) for row in rows
    )
    adaptive_budget_local_llm_call_count_count_total = sum(
        int(row.get("adaptive_budget_local_llm_call_count_count") or 0) for row in rows
    )
    adaptive_budget_reward_progress_score_sum_total = sum(
        float(row.get("adaptive_budget_reward_progress_score_sum") or 0.0) for row in rows
    )
    adaptive_budget_reward_progress_score_count_total = sum(
        int(row.get("adaptive_budget_reward_progress_score_count") or 0) for row in rows
    )
    adaptive_budget_propose_budget_change_total = sum(
        int(row.get("adaptive_budget_propose_budget_change_attempts") or 0) for row in rows
    )
    adaptive_budget_max_tokens_change_total = sum(
        int(row.get("adaptive_budget_max_tokens_change_attempts") or 0) for row in rows
    )
    adaptive_budget_helper_allowance_change_total = sum(
        int(row.get("adaptive_budget_helper_allowance_change_attempts") or 0) for row in rows
    )
    adaptive_budget_stage_a_requests_change_total = sum(
        int(row.get("adaptive_budget_stage_a_requests_change_attempts") or 0) for row in rows
    )
    adaptive_budget_stage_a_chars_change_total = sum(
        int(row.get("adaptive_budget_stage_a_chars_change_attempts") or 0) for row in rows
    )
    adaptive_budget_retry_history_budget_change_total = sum(
        int(row.get("adaptive_budget_retry_history_budget_change_attempts") or 0) for row in rows
    )
    adaptive_budget_compaction_allowance_change_total = sum(
        int(row.get("adaptive_budget_compaction_allowance_change_attempts") or 0) for row in rows
    )
    adaptive_budget_response_mode_change_total = sum(
        int(row.get("adaptive_budget_response_mode_change_attempts") or 0) for row in rows
    )
    adaptive_budget_partial_override_total = sum(
        int(row.get("adaptive_budget_partial_override_attempts") or 0) for row in rows
    )
    adaptive_budget_effective_helper_max_calls_sum_total = sum(
        float(row.get("adaptive_budget_effective_helper_max_calls_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_helper_max_calls_count_total = sum(
        int(row.get("adaptive_budget_effective_helper_max_calls_count") or 0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_requests_sum_total = sum(
        float(row.get("adaptive_budget_effective_stage_a_max_requests_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_requests_count_total = sum(
        int(row.get("adaptive_budget_effective_stage_a_max_requests_count") or 0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_chars_sum_total = sum(
        float(row.get("adaptive_budget_effective_stage_a_max_chars_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_chars_count_total = sum(
        int(row.get("adaptive_budget_effective_stage_a_max_chars_count") or 0) for row in rows
    )
    adaptive_budget_effective_retry_history_max_chars_sum_total = sum(
        float(row.get("adaptive_budget_effective_retry_history_max_chars_sum") or 0.0)
        for row in rows
    )
    adaptive_budget_effective_retry_history_max_chars_count_total = sum(
        int(row.get("adaptive_budget_effective_retry_history_max_chars_count") or 0) for row in rows
    )
    adaptive_budget_effective_context_compaction_max_calls_sum_total = sum(
        float(row.get("adaptive_budget_effective_context_compaction_max_calls_sum") or 0.0)
        for row in rows
    )
    adaptive_budget_effective_context_compaction_max_calls_count_total = sum(
        int(row.get("adaptive_budget_effective_context_compaction_max_calls_count") or 0)
        for row in rows
    )
    adaptive_budget_effective_openai_max_tokens_sum_total = sum(
        float(row.get("adaptive_budget_effective_openai_max_tokens_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_openai_max_tokens_count_total = sum(
        int(row.get("adaptive_budget_effective_openai_max_tokens_count") or 0) for row in rows
    )
    syntax_preflight_enabled_attempts_total = sum(
        int(row.get("syntax_preflight_enabled_attempts") or 0) for row in rows
    )
    syntax_preflight_cycles_sum_total = sum(
        int(row.get("syntax_preflight_cycles_sum") or 0) for row in rows
    )
    syntax_preflight_failures_sum_total = sum(
        int(row.get("syntax_preflight_failures_sum") or 0) for row in rows
    )
    syntax_preflight_failed_attempts_total = sum(
        int(row.get("syntax_preflight_failed_attempts") or 0) for row in rows
    )
    patch_verifier_attempts_total = sum(
        int(row.get("patch_verifier_attempts") or 0) for row in rows
    )
    patch_verifier_source_scope_matches_total = sum(
        int(row.get("patch_verifier_source_scope_matches") or 0) for row in rows
    )
    patch_verifier_recovery_eligible_attempts_total = sum(
        int(row.get("patch_verifier_recovery_eligible_attempts") or 0) for row in rows
    )
    patch_verifier_recovery_invoked_attempts_total = sum(
        int(row.get("patch_verifier_recovery_invoked_attempts") or 0) for row in rows
    )
    patch_verifier_recovery_passed_attempts_total = sum(
        int(row.get("patch_verifier_recovery_passed_attempts") or 0) for row in rows
    )
    patch_verifier_recovery_result_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        result_counts = row.get("patch_verifier_recovery_result_counts")
        if not isinstance(result_counts, dict):
            continue
        for result, value in result_counts.items():
            if isinstance(result, str) and isinstance(value, (int, float)):
                patch_verifier_recovery_result_counts_total[result] += int(value)
    patch_verifier_recovery_terminal_note_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        note_counts = row.get("patch_verifier_recovery_terminal_note_counts")
        if not isinstance(note_counts, dict):
            continue
        for note, value in note_counts.items():
            if isinstance(note, str) and isinstance(value, (int, float)):
                patch_verifier_recovery_terminal_note_counts_total[note] += int(value)

    scoped_records = _records_for_rows(rows, records_by_case_condition=records_by_case_condition)
    adaptive_online_ope = _adaptive_ope_summary_from_records(
        records=scoped_records,
        prefix="adaptive_online",
        action_space=ADAPTIVE_ONLINE_ACTIONS,
        target_policies=ADAPTIVE_ONLINE_TARGET_POLICIES,
        min_action_support=adaptive_ope_min_action_support,
        propensity_floor=adaptive_ope_propensity_floor,
    )
    adaptive_budget_ope = _adaptive_ope_summary_from_records(
        records=scoped_records,
        prefix="adaptive_budget",
        action_space=ADAPTIVE_BUDGET_ACTIONS,
        target_policies=ADAPTIVE_BUDGET_TARGET_POLICIES,
        min_action_support=adaptive_ope_min_action_support,
        propensity_floor=adaptive_ope_propensity_floor,
    )
    adaptive_online_reward_std = _stddev(adaptive_online_reward_values_total)
    adaptive_online_reward_min = min(adaptive_online_reward_values_total, default=None)
    adaptive_online_reward_p50 = _quantile(adaptive_online_reward_values_total, 0.5)
    adaptive_online_reward_p90 = _quantile(adaptive_online_reward_values_total, 0.9)
    adaptive_online_reward_max = max(adaptive_online_reward_values_total, default=None)
    adaptive_budget_reward_std = _stddev(adaptive_budget_reward_values_total)
    adaptive_budget_reward_min = min(adaptive_budget_reward_values_total, default=None)
    adaptive_budget_reward_p50 = _quantile(adaptive_budget_reward_values_total, 0.5)
    adaptive_budget_reward_p90 = _quantile(adaptive_budget_reward_values_total, 0.9)
    adaptive_budget_reward_max = max(adaptive_budget_reward_values_total, default=None)

    out = {
        "helper_invoked_cases": helper_invoked_cases,
        "helper_invocation_case_rate": ((helper_invoked_cases / len(rows)) if rows else None),
        "helper_invocation_attempts": helper_invocation_attempts_total,
        "helper_invocation_attempt_rate": (
            (helper_invocation_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_materialized_cases": helper_materialized_cases,
        "helper_materialized_case_rate": (
            (helper_materialized_cases / len(rows)) if rows else None
        ),
        "helper_materialized_attempts": helper_materialized_attempts_total,
        "helper_materialized_attempt_rate": (
            (helper_materialized_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_low_yield_attempts": helper_low_yield_attempts_total,
        "helper_low_yield_attempt_rate": (
            (helper_low_yield_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_transport_failure_attempts": helper_transport_failure_attempts_total,
        "helper_transport_failure_attempt_rate": (
            (helper_transport_failure_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_backend_unavailable_attempts": helper_backend_unavailable_attempts_total,
        "helper_backend_unavailable_attempt_rate": (
            (helper_backend_unavailable_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_compat_reasoning_only_attempts": helper_compat_reasoning_only_attempts_total,
        "helper_compat_reasoning_only_attempt_rate": (
            (helper_compat_reasoning_only_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_invalid_payload_attempts": helper_invalid_payload_attempts_total,
        "helper_invalid_payload_attempt_rate": (
            (helper_invalid_payload_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_request_budget_exceeded_attempts": helper_request_budget_exceeded_attempts_total,
        "helper_request_budget_exceeded_attempt_rate": (
            (helper_request_budget_exceeded_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_retry_attempts": helper_retry_attempts_total,
        "helper_transport_retry_attempts": helper_transport_retry_attempts_total,
        "helper_mean_retries_per_invocation": (
            (helper_retry_attempts_total / helper_invocation_attempts_total)
            if helper_invocation_attempts_total > 0
            else None
        ),
        "helper_mean_transport_retries_per_invocation": (
            (helper_transport_retry_attempts_total / helper_invocation_attempts_total)
            if helper_invocation_attempts_total > 0
            else None
        ),
        "helper_effective_attempts": helper_effective_attempts_total,
        "helper_effective_success_rate": (
            (helper_materialized_attempts_total / helper_effective_attempts_total)
            if helper_effective_attempts_total > 0
            else None
        ),
        "helper_success_when_invoked": (
            (helper_success_with_invocation_cases / helper_invoked_cases)
            if helper_invoked_cases > 0
            else None
        ),
        "helper_rescue_rate": (
            (helper_rescued_cases / helper_invoked_cases) if helper_invoked_cases > 0 else None
        ),
        "helper_decision_attempts": helper_decision_attempts_total,
        "helper_need_more_true_rate": (
            (helper_need_more_true_attempts_total / helper_need_more_labeled_attempts)
            if helper_need_more_labeled_attempts > 0
            else None
        ),
        "helper_budget_exhausted_rate": (
            (helper_budget_exhausted_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_saturation_guard_rate": (
            (helper_saturation_guard_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "adaptive_budget_topup_attempts": adaptive_budget_topup_attempts_total,
        "adaptive_budget_topup_attempt_rate": (
            (adaptive_budget_topup_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "adaptive_budget_topup_cases": adaptive_budget_topup_cases,
        "adaptive_budget_topup_case_rate": (
            (adaptive_budget_topup_cases / len(rows)) if rows else None
        ),
        "adaptive_online_decisions": adaptive_online_decisions_total,
        "adaptive_online_selected_action_counts": {
            action: int(adaptive_online_selected_action_counts_total.get(action, 0))
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_selected_action_rates": {
            action: (
                adaptive_online_selected_action_counts_total.get(action, 0)
                / adaptive_online_decisions_total
            )
            if adaptive_online_decisions_total > 0
            else None
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_applied_action_counts": {
            action: int(adaptive_online_applied_action_counts_total.get(action, 0))
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_applied_action_rates": {
            action: (
                adaptive_online_applied_action_counts_total.get(action, 0)
                / adaptive_online_decisions_total
            )
            if adaptive_online_decisions_total > 0
            else None
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_behavior_action_counts": {
            action: int(adaptive_online_behavior_action_counts_total.get(action, 0))
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_behavior_action_rates": {
            action: (
                adaptive_online_behavior_action_counts_total.get(action, 0)
                / adaptive_online_decisions_total
            )
            if adaptive_online_decisions_total > 0
            else None
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_selected_invoke_attempts": adaptive_online_selected_invoke_total,
        "adaptive_online_selected_invoke_rate": (
            (adaptive_online_selected_invoke_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_applied_invoke_attempts": adaptive_online_applied_invoke_total,
        "adaptive_online_applied_invoke_rate": (
            (adaptive_online_applied_invoke_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_override_attempts": adaptive_online_override_total,
        "adaptive_online_override_rate": (
            (adaptive_online_override_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_updates": adaptive_online_updates_total,
        "adaptive_online_update_rate": (
            (adaptive_online_updates_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_mean_reward": (
            (adaptive_online_reward_sum_total / adaptive_online_reward_count_total)
            if adaptive_online_reward_count_total > 0
            else None
        ),
        "adaptive_online_reward_std": adaptive_online_reward_std,
        "adaptive_online_reward_min": adaptive_online_reward_min,
        "adaptive_online_reward_p50": adaptive_online_reward_p50,
        "adaptive_online_reward_p90": adaptive_online_reward_p90,
        "adaptive_online_reward_max": adaptive_online_reward_max,
        "adaptive_online_mean_propensity": (
            (adaptive_online_propensity_sum_total / adaptive_online_propensity_count_total)
            if adaptive_online_propensity_count_total > 0
            else None
        ),
        "adaptive_online_mean_local_tokens_total": (
            (
                adaptive_online_local_tokens_total_sum_total
                / adaptive_online_local_tokens_total_count_total
            )
            if adaptive_online_local_tokens_total_count_total > 0
            else None
        ),
        "adaptive_online_mean_local_wall_sec": (
            (adaptive_online_local_wall_sec_sum_total / adaptive_online_local_wall_sec_count_total)
            if adaptive_online_local_wall_sec_count_total > 0
            else None
        ),
        "adaptive_online_mean_local_llm_call_count": (
            (
                adaptive_online_local_llm_call_count_sum_total
                / adaptive_online_local_llm_call_count_count_total
            )
            if adaptive_online_local_llm_call_count_count_total > 0
            else None
        ),
        "adaptive_online_mean_progress_score": (
            (
                adaptive_online_reward_progress_score_sum_total
                / adaptive_online_reward_progress_score_count_total
            )
            if adaptive_online_reward_progress_score_count_total > 0
            else None
        ),
        "adaptive_online_progress_reason_counts": dict(
            sorted(adaptive_online_progress_reason_counts_total.items())
        ),
        "adaptive_online_delayed_rescue_bonus_sum": adaptive_online_delayed_rescue_bonus_sum_total,
        "adaptive_online_delayed_rescue_bonus_count": (
            adaptive_online_delayed_rescue_bonus_count_total
        ),
        "adaptive_online_delayed_rescue_bonus_mean": (
            (
                adaptive_online_delayed_rescue_bonus_sum_total
                / adaptive_online_delayed_rescue_bonus_count_total
            )
            if adaptive_online_delayed_rescue_bonus_count_total > 0
            else None
        ),
        "adaptive_online_credited_rescues_plus_1": adaptive_online_credited_rescues_plus_1_total,
        "adaptive_online_credited_rescues_plus_2": adaptive_online_credited_rescues_plus_2_total,
        "adaptive_budget_decisions": adaptive_budget_decisions_total,
        "adaptive_budget_selected_increase_attempts": adaptive_budget_selected_increase_total,
        "adaptive_budget_selected_increase_rate": (
            (adaptive_budget_selected_increase_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_increase_attempts": adaptive_budget_applied_increase_total,
        "adaptive_budget_applied_increase_rate": (
            (adaptive_budget_applied_increase_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_selected_time_attempts": adaptive_budget_selected_time_total,
        "adaptive_budget_selected_time_rate": (
            (adaptive_budget_selected_time_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_selected_tokens_attempts": adaptive_budget_selected_tokens_total,
        "adaptive_budget_selected_tokens_rate": (
            (adaptive_budget_selected_tokens_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_selected_context_attempts": adaptive_budget_selected_context_total,
        "adaptive_budget_selected_context_rate": (
            (adaptive_budget_selected_context_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_time_attempts": adaptive_budget_applied_time_total,
        "adaptive_budget_applied_time_rate": (
            (adaptive_budget_applied_time_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_tokens_attempts": adaptive_budget_applied_tokens_total,
        "adaptive_budget_applied_tokens_rate": (
            (adaptive_budget_applied_tokens_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_context_attempts": adaptive_budget_applied_context_total,
        "adaptive_budget_applied_context_rate": (
            (adaptive_budget_applied_context_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_override_attempts": adaptive_budget_override_total,
        "adaptive_budget_override_rate": (
            (adaptive_budget_override_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_propose_budget_change_attempts": adaptive_budget_propose_budget_change_total,
        "adaptive_budget_propose_budget_change_rate": (
            (adaptive_budget_propose_budget_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_max_tokens_change_attempts": adaptive_budget_max_tokens_change_total,
        "adaptive_budget_max_tokens_change_rate": (
            (adaptive_budget_max_tokens_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_helper_allowance_change_attempts": (
            adaptive_budget_helper_allowance_change_total
        ),
        "adaptive_budget_helper_allowance_change_rate": (
            (adaptive_budget_helper_allowance_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_stage_a_requests_change_attempts": (
            adaptive_budget_stage_a_requests_change_total
        ),
        "adaptive_budget_stage_a_requests_change_rate": (
            (adaptive_budget_stage_a_requests_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_stage_a_chars_change_attempts": adaptive_budget_stage_a_chars_change_total,
        "adaptive_budget_stage_a_chars_change_rate": (
            (adaptive_budget_stage_a_chars_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_retry_history_budget_change_attempts": (
            adaptive_budget_retry_history_budget_change_total
        ),
        "adaptive_budget_retry_history_budget_change_rate": (
            (adaptive_budget_retry_history_budget_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_compaction_allowance_change_attempts": (
            adaptive_budget_compaction_allowance_change_total
        ),
        "adaptive_budget_compaction_allowance_change_rate": (
            (adaptive_budget_compaction_allowance_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_response_mode_change_attempts": adaptive_budget_response_mode_change_total,
        "adaptive_budget_response_mode_change_rate": (
            (adaptive_budget_response_mode_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_partial_override_attempts": adaptive_budget_partial_override_total,
        "adaptive_budget_partial_override_rate": (
            (adaptive_budget_partial_override_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_updates": adaptive_budget_updates_total,
        "adaptive_budget_update_rate": (
            (adaptive_budget_updates_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_mean_reward": (
            (adaptive_budget_reward_sum_total / adaptive_budget_reward_count_total)
            if adaptive_budget_reward_count_total > 0
            else None
        ),
        "adaptive_budget_reward_std": adaptive_budget_reward_std,
        "adaptive_budget_reward_min": adaptive_budget_reward_min,
        "adaptive_budget_reward_p50": adaptive_budget_reward_p50,
        "adaptive_budget_reward_p90": adaptive_budget_reward_p90,
        "adaptive_budget_reward_max": adaptive_budget_reward_max,
        "adaptive_budget_mean_propensity": (
            (adaptive_budget_propensity_sum_total / adaptive_budget_propensity_count_total)
            if adaptive_budget_propensity_count_total > 0
            else None
        ),
        "adaptive_budget_mean_family_propensity": (
            (
                adaptive_budget_family_propensity_sum_total
                / adaptive_budget_family_propensity_count_total
            )
            if adaptive_budget_family_propensity_count_total > 0
            else None
        ),
        "adaptive_budget_mean_local_tokens_total": (
            (
                adaptive_budget_local_tokens_total_sum_total
                / adaptive_budget_local_tokens_total_count_total
            )
            if adaptive_budget_local_tokens_total_count_total > 0
            else None
        ),
        "adaptive_budget_mean_local_wall_sec": (
            (adaptive_budget_local_wall_sec_sum_total / adaptive_budget_local_wall_sec_count_total)
            if adaptive_budget_local_wall_sec_count_total > 0
            else None
        ),
        "adaptive_budget_mean_local_llm_call_count": (
            (
                adaptive_budget_local_llm_call_count_sum_total
                / adaptive_budget_local_llm_call_count_count_total
            )
            if adaptive_budget_local_llm_call_count_count_total > 0
            else None
        ),
        "adaptive_budget_mean_progress_score": (
            (
                adaptive_budget_reward_progress_score_sum_total
                / adaptive_budget_reward_progress_score_count_total
            )
            if adaptive_budget_reward_progress_score_count_total > 0
            else None
        ),
        "adaptive_budget_request_budget_exceeded_attempts": adaptive_budget_request_budget_exceeded_total,
        "adaptive_budget_request_budget_exceeded_rate": (
            (adaptive_budget_request_budget_exceeded_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "adaptive_budget_mean_applied_multiplier": (
            (
                adaptive_budget_applied_multiplier_sum_total
                / adaptive_budget_applied_multiplier_count_total
            )
            if adaptive_budget_applied_multiplier_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_helper_max_calls": (
            (
                adaptive_budget_effective_helper_max_calls_sum_total
                / adaptive_budget_effective_helper_max_calls_count_total
            )
            if adaptive_budget_effective_helper_max_calls_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_stage_a_max_requests": (
            (
                adaptive_budget_effective_stage_a_max_requests_sum_total
                / adaptive_budget_effective_stage_a_max_requests_count_total
            )
            if adaptive_budget_effective_stage_a_max_requests_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_stage_a_max_chars": (
            (
                adaptive_budget_effective_stage_a_max_chars_sum_total
                / adaptive_budget_effective_stage_a_max_chars_count_total
            )
            if adaptive_budget_effective_stage_a_max_chars_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_retry_history_max_chars": (
            (
                adaptive_budget_effective_retry_history_max_chars_sum_total
                / adaptive_budget_effective_retry_history_max_chars_count_total
            )
            if adaptive_budget_effective_retry_history_max_chars_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_context_compaction_max_calls": (
            (
                adaptive_budget_effective_context_compaction_max_calls_sum_total
                / adaptive_budget_effective_context_compaction_max_calls_count_total
            )
            if adaptive_budget_effective_context_compaction_max_calls_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_openai_max_tokens": (
            (
                adaptive_budget_effective_openai_max_tokens_sum_total
                / adaptive_budget_effective_openai_max_tokens_count_total
            )
            if adaptive_budget_effective_openai_max_tokens_count_total > 0
            else None
        ),
        "adaptive_online_ope": adaptive_online_ope,
        "adaptive_budget_ope": adaptive_budget_ope,
        "mean_syntax_preflight_cycles": (
            (syntax_preflight_cycles_sum_total / syntax_preflight_enabled_attempts_total)
            if syntax_preflight_enabled_attempts_total > 0
            else None
        ),
        "syntax_preflight_failures": syntax_preflight_failures_sum_total,
        "syntax_preflight_failed_attempts": syntax_preflight_failed_attempts_total,
        "syntax_preflight_failed_attempt_rate": (
            (syntax_preflight_failed_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "patch_verifier_source_scope_matches": patch_verifier_source_scope_matches_total,
        "patch_verifier_source_scope_match_rate": (
            (patch_verifier_source_scope_matches_total / patch_verifier_attempts_total)
            if patch_verifier_attempts_total > 0
            else None
        ),
        "patch_verifier_recovery_eligible_attempts": (
            patch_verifier_recovery_eligible_attempts_total
        ),
        "patch_verifier_recovery_invoked_attempts": patch_verifier_recovery_invoked_attempts_total,
        "patch_verifier_recovery_passed_attempts": patch_verifier_recovery_passed_attempts_total,
        "patch_verifier_recovery_pass_rate": (
            (
                patch_verifier_recovery_passed_attempts_total
                / patch_verifier_recovery_invoked_attempts_total
            )
            if patch_verifier_recovery_invoked_attempts_total > 0
            else None
        ),
        "patch_verifier_recovery_result_counts": dict(
            sorted(patch_verifier_recovery_result_counts_total.items())
        ),
        "patch_verifier_recovery_terminal_note_counts": dict(
            sorted(patch_verifier_recovery_terminal_note_counts_total.items())
        ),
    }
    if include_condition_only_counts:
        out.update(
            {
                "helper_success_with_invocation_cases": helper_success_with_invocation_cases,
                "helper_rescued_cases": helper_rescued_cases,
                "mean_helper_calls_used_final": _mean(helper_calls_used_values),
                "helper_materialized_attempts": helper_materialized_attempts_total,
                "helper_low_yield_attempts": helper_low_yield_attempts_total,
                "helper_need_more_true_attempts": helper_need_more_true_attempts_total,
                "helper_need_more_false_attempts": helper_need_more_false_attempts_total,
                "helper_budget_exhausted_attempts": helper_budget_exhausted_attempts_total,
                "helper_saturation_guard_attempts": helper_saturation_guard_attempts_total,
            }
        )
    return out


def _records_for_rows(
    rows: list[dict[str, Any]],
    *,
    records_by_case_condition: dict[tuple[str, str, str], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Collect raw records corresponding to the given derived rows."""
    if not records_by_case_condition:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("run_id") or ""),
            str(row.get("case_id") or ""),
            str(row.get("condition") or ""),
        )
        out.extend(records_by_case_condition.get(key, []))
    return out


def _pairwise_uplifts(*, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute pairwise success-rate uplifts for all condition pairs."""
    conditions = sorted(
        condition
        for condition, stats in summary.items()
        if isinstance(stats, dict) and "cases" in stats and "success_rate" in stats
    )
    out: list[dict[str, Any]] = []
    for idx, condition_a in enumerate(conditions):
        for condition_b in conditions[idx + 1 :]:
            uplift = _uplift_between_conditions(
                summary=summary,
                a_condition=condition_a,
                b_condition=condition_b,
            )
            out.append(
                {
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "uplift_success_rate": uplift,
                }
            )
    return out


def _context_protocol_from_records(records: list[dict[str, Any]]) -> str:
    """Extract the context protocol label, defaulting to ``'monolithic'``."""
    for record in records:
        protocol = record.get("context_protocol")
        if isinstance(protocol, str) and protocol:
            return protocol
        attempt_meta = record.get("attempt_metadata")
        if isinstance(attempt_meta, dict):
            maybe = attempt_meta.get("context_protocol")
            if isinstance(maybe, str) and maybe:
                return maybe
    return "monolithic"


def _stage_b_prompt_chars_mean(records: list[dict[str, Any]]) -> float | None:
    """Mean stage-B prompt character count across attempt records."""
    values: list[float] = []
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        value = attempt_meta.get("stage_b_prompt_chars")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return _mean(values)


def _stage_a_request_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count stage-A request attempts and successes.

    Returns:
        Tuple of (total_attempts, total_successes).
    """
    attempts = 0
    successes = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        success = attempt_meta.get("stage_a_request_success")
        if success is None:
            continue
        attempts += 1
        if bool(success):
            successes += 1
    return attempts, successes


def _retry_gate_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count retry-gate evaluations and blocks across records.

    Returns:
        Tuple of (evaluated_attempts, blocked_attempts).
    """
    attempts = 0
    blocks = 0
    for record in records:
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        # ATT-05: unscored blocks are infra events; the next scored attempt counts them.
        blocks_before = patch_meta.get("retry_gate_blocks_before")
        if isinstance(blocks_before, int) and blocks_before > 0:
            attempts += blocks_before
            blocks += blocks_before
        evaluated = patch_meta.get("retry_gate_evaluated")
        blocked = patch_meta.get("retry_gate_blocked")
        if isinstance(evaluated, bool):
            attempts += 1
            if bool(blocked):
                blocks += 1
            continue
        if blocked is None:
            continue
        attempts += 1
        if bool(blocked):
            blocks += 1
    return attempts, blocks


def _adaptive_budget_topup_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count attempts where an adaptive budget top-up was granted."""
    attempts = 0
    for record in records:
        attempt = _as_int(record.get("attempt"))
        if attempt <= 0:
            continue
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
        granted = attempt_meta_dict.get(
            "adaptive_budget_topup_granted",
            patch_meta_dict.get("adaptive_budget_topup_granted"),
        )
        if bool(granted):
            attempts += 1
    return {"topup_attempts": attempts}


def _syntax_preflight_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate syntax-preflight cycle, failure, and enablement metrics."""
    enabled_attempts = 0
    cycles_sum = 0
    failures_sum = 0
    failed_attempts = 0
    for record in records:
        attempt = _as_int(record.get("attempt"))
        if attempt <= 0:
            continue
        note = str(record.get("note") or "")
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
        enabled = bool(
            attempt_meta_dict.get(
                "syntax_preflight_enabled",
                patch_meta_dict.get("syntax_preflight_enabled"),
            )
        )
        if enabled:
            enabled_attempts += 1
        cycles_value = attempt_meta_dict.get(
            "syntax_preflight_cycles_used",
            patch_meta_dict.get("syntax_preflight_cycles_used"),
        )
        failures_value = attempt_meta_dict.get(
            "syntax_preflight_failures",
            patch_meta_dict.get("syntax_preflight_failures"),
        )
        if isinstance(cycles_value, (int, float)):
            cycles_sum += int(cycles_value)
        if isinstance(failures_value, (int, float)):
            failures_sum += int(failures_value)
        if note == "syntax_preflight_failed":
            failed_attempts += 1
    return {
        "enabled_attempts": enabled_attempts,
        "cycles_sum": cycles_sum,
        "failures_sum": failures_sum,
        "failed_attempts": failed_attempts,
    }


def _patch_verifier_source_scope_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate patch-verifier source-scope and recovery metrics.

    Counts verifier invocations, source-scope matches, recovery
    eligibility, invocations, pass/fail results, and terminal notes.
    """
    verifier_attempts = 0
    source_scope_matches = 0
    recovery_eligible_attempts = 0
    recovery_invoked_attempts = 0
    recovery_passed_attempts = 0
    recovery_result_counts: dict[str, int] = defaultdict(int)
    recovery_terminal_note_counts: dict[str, int] = defaultdict(int)
    for record in records:
        attempt = _as_int(record.get("attempt"))
        if attempt <= 0:
            continue
        note = str(record.get("note") or "")
        attempt_meta = record.get("attempt_metadata")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta = record.get("patch_meta")
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
        if not patch_meta_dict:
            continue
        if not bool(patch_meta_dict.get("patch_verifier_enabled")):
            continue
        verifier_attempts += 1
        if bool(patch_meta_dict.get("patch_verifier_source_scope_matched")):
            source_scope_matches += 1
        eligible = bool(
            attempt_meta_dict.get(
                "patch_verifier_recovery_eligible",
                patch_meta_dict.get("patch_verifier_recovery_eligible"),
            )
        )
        invoked = bool(
            attempt_meta_dict.get(
                "patch_verifier_recovery_invoked",
                patch_meta_dict.get("patch_verifier_recovery_invoked"),
            )
        )
        result = str(
            attempt_meta_dict.get(
                "patch_verifier_recovery_result",
                patch_meta_dict.get("patch_verifier_recovery_result"),
            )
            or ""
        ).strip()
        if eligible:
            recovery_eligible_attempts += 1
        if invoked:
            recovery_invoked_attempts += 1
            if note:
                recovery_terminal_note_counts[note] += 1
        if invoked and result == "passed_verifier":
            recovery_passed_attempts += 1
        if (
            eligible
            or invoked
            or (patch_meta_dict.get("patch_verifier_passed") is False and result)
        ):
            recovery_result_counts[result or "not_eligible"] += 1
    return {
        "verifier_attempts": verifier_attempts,
        "source_scope_matches": source_scope_matches,
        "recovery_eligible_attempts": recovery_eligible_attempts,
        "recovery_invoked_attempts": recovery_invoked_attempts,
        "recovery_passed_attempts": recovery_passed_attempts,
        "recovery_result_counts": dict(sorted(recovery_result_counts.items())),
        "recovery_terminal_note_counts": dict(sorted(recovery_terminal_note_counts.items())),
    }


def _adaptive_helper_outcome(
    *,
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
) -> str:
    """Classify the adaptive helper outcome for a single attempt.

    Returns one of: ``'materialized'``, ``'transport_failed'``,
    ``'unavailable_backend'``, ``'compat_reasoning_only'``,
    ``'request_budget_exceeded'``,
    ``'invalid_payload'``, ``'semantic_no_context'``,
    ``'skipped_unsupported'``, ``'skipped_guardrail'``,
    ``'skipped_policy'``, or ``'not_invoked'``.
    """
    explicit = attempt_meta_dict.get(
        "adaptive_helper_outcome",
        patch_meta_dict.get("adaptive_helper_outcome"),
    )
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip()
        if normalized == "backend_unavailable":
            return "unavailable_backend"
        return normalized

    failure_kind_raw = attempt_meta_dict.get(
        "adaptive_helper_failure_kind",
        patch_meta_dict.get("adaptive_helper_failure_kind"),
    )
    failure_kind = str(failure_kind_raw or "").strip()

    helper_invoked_raw = attempt_meta_dict.get(
        "adaptive_helper_invoked",
        patch_meta_dict.get("adaptive_helper_invoked"),
    )
    helper_invoked = bool(helper_invoked_raw) if isinstance(helper_invoked_raw, bool) else False
    helper_reason = str(
        attempt_meta_dict.get(
            "adaptive_helper_decision_reason",
            patch_meta_dict.get("adaptive_helper_decision_reason"),
        )
        or ""
    ).strip()
    helper_materialized_raw = attempt_meta_dict.get(
        "adaptive_helper_materialized_context",
        patch_meta_dict.get("adaptive_helper_materialized_context"),
    )
    helper_materialized = (
        bool(helper_materialized_raw) if isinstance(helper_materialized_raw, bool) else False
    )
    if helper_invoked:
        if helper_materialized:
            return "materialized"
        if failure_kind == "transport":
            return "transport_failed"
        if failure_kind == "backend_unavailable":
            return "unavailable_backend"
        if failure_kind == "compat_reasoning_only":
            return "compat_reasoning_only"
        if failure_kind == "request_budget_exceeded":
            return "request_budget_exceeded"
        if failure_kind == "invalid_payload":
            return "invalid_payload"
        if helper_reason == "llm_invalid_response_no_helper":
            return "invalid_payload"
        return "semantic_no_context"
    if helper_reason == "request_context_unavailable":
        return "skipped_unsupported"
    if helper_reason in {
        "informativeness_pregate_skip",
        "helper_budget_exhausted",
        "helper_saturation_guard",
    }:
        return "skipped_guardrail"
    if helper_reason in {"not_adaptive_condition", ""}:
        return "not_invoked"
    return "skipped_policy"


def _helper_retry_count(
    *, attempt_meta_dict: dict[str, Any], patch_meta_dict: dict[str, Any]
) -> int:
    """Extract the helper retry count for a single attempt."""
    value = attempt_meta_dict.get(
        "adaptive_helper_retry_count",
        patch_meta_dict.get("adaptive_helper_retry_count"),
    )
    return _as_int(value)


def _helper_transport_retry_count(
    *,
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
) -> int:
    """Extract the helper transport retry count for a single attempt."""
    value = attempt_meta_dict.get(
        "adaptive_helper_transport_retry_count",
        patch_meta_dict.get("adaptive_helper_transport_retry_count"),
    )
    return _as_int(value)


def _adaptive_helper_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate adaptive helper invocation and outcome metrics.

    Walks all attempt records to tally invocations, materialization,
    transport failures, retries, budget exhaustion, and saturation
    guard activations.
    """
    helper_invocation_attempts = 0
    helper_materialized_attempts = 0
    helper_low_yield_attempts = 0
    helper_transport_failure_attempts = 0
    helper_backend_unavailable_attempts = 0
    helper_compat_reasoning_only_attempts = 0
    helper_invalid_payload_attempts = 0
    helper_request_budget_exceeded_attempts = 0
    helper_retry_attempts = 0
    helper_transport_retry_attempts = 0
    helper_calls_used_final = 0
    helper_decision_attempts = 0
    helper_need_more_true_attempts = 0
    helper_need_more_false_attempts = 0
    helper_budget_exhausted_attempts = 0
    helper_saturation_guard_attempts = 0
    first_helper_attempt: int | None = None
    adaptive_mode = "none"

    for record in records:
        attempt = _as_int(record.get("attempt"))
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        mode_raw = attempt_meta_dict.get("adaptive_mode", patch_meta_dict.get("adaptive_mode"))
        if isinstance(mode_raw, str) and mode_raw and adaptive_mode == "none":
            adaptive_mode = mode_raw
        if attempt > 0 and mode_raw in {"gated", "llm_discretion"}:
            helper_decision_attempts += 1

        helper_invoked_raw = attempt_meta_dict.get(
            "adaptive_helper_invoked", patch_meta_dict.get("adaptive_helper_invoked")
        )
        if isinstance(helper_invoked_raw, bool) and helper_invoked_raw:
            helper_invocation_attempts += 1
            if first_helper_attempt is None:
                first_helper_attempt = attempt
        helper_outcome = _adaptive_helper_outcome(
            attempt_meta_dict=attempt_meta_dict,
            patch_meta_dict=patch_meta_dict,
        )
        if helper_outcome == "materialized":
            helper_materialized_attempts += 1
        if helper_outcome == "semantic_no_context":
            helper_low_yield_attempts += 1
        if helper_outcome == "transport_failed":
            helper_transport_failure_attempts += 1
        if helper_outcome == "unavailable_backend":
            helper_backend_unavailable_attempts += 1
        if helper_outcome == "compat_reasoning_only":
            helper_compat_reasoning_only_attempts += 1
        if helper_outcome == "invalid_payload":
            helper_invalid_payload_attempts += 1
        if helper_outcome == "request_budget_exceeded":
            helper_request_budget_exceeded_attempts += 1

        helper_retry_attempts += _helper_retry_count(
            attempt_meta_dict=attempt_meta_dict,
            patch_meta_dict=patch_meta_dict,
        )
        helper_transport_retry_attempts += _helper_transport_retry_count(
            attempt_meta_dict=attempt_meta_dict,
            patch_meta_dict=patch_meta_dict,
        )

        helper_calls_after = attempt_meta_dict.get(
            "adaptive_helper_calls_used_after_attempt",
            patch_meta_dict.get("adaptive_helper_calls_used_after_attempt"),
        )
        if isinstance(helper_calls_after, (int, float)):
            helper_calls_used_final = max(helper_calls_used_final, int(helper_calls_after))

        need_more_context_raw = attempt_meta_dict.get(
            "adaptive_helper_need_more_context",
            patch_meta_dict.get("adaptive_helper_need_more_context"),
        )
        if isinstance(need_more_context_raw, bool):
            if need_more_context_raw:
                helper_need_more_true_attempts += 1
            else:
                helper_need_more_false_attempts += 1

        if bool(
            attempt_meta_dict.get(
                "adaptive_helper_budget_exhausted",
                patch_meta_dict.get("adaptive_helper_budget_exhausted"),
            )
        ):
            helper_budget_exhausted_attempts += 1
        if bool(
            attempt_meta_dict.get(
                "adaptive_helper_saturation_guarded",
                patch_meta_dict.get("adaptive_helper_saturation_guarded"),
            )
        ):
            helper_saturation_guard_attempts += 1

    return {
        "adaptive_mode": adaptive_mode,
        "helper_invocation_attempts": helper_invocation_attempts,
        "helper_materialized_attempts": helper_materialized_attempts,
        "helper_low_yield_attempts": helper_low_yield_attempts,
        "helper_transport_failure_attempts": helper_transport_failure_attempts,
        "helper_backend_unavailable_attempts": helper_backend_unavailable_attempts,
        "helper_compat_reasoning_only_attempts": helper_compat_reasoning_only_attempts,
        "helper_invalid_payload_attempts": helper_invalid_payload_attempts,
        "helper_request_budget_exceeded_attempts": helper_request_budget_exceeded_attempts,
        "helper_retry_attempts": helper_retry_attempts,
        "helper_transport_retry_attempts": helper_transport_retry_attempts,
        "helper_calls_used_final": helper_calls_used_final,
        "helper_decision_attempts": helper_decision_attempts,
        "helper_need_more_true_attempts": helper_need_more_true_attempts,
        "helper_need_more_false_attempts": helper_need_more_false_attempts,
        "helper_budget_exhausted_attempts": helper_budget_exhausted_attempts,
        "helper_saturation_guard_attempts": helper_saturation_guard_attempts,
        "first_helper_attempt": first_helper_attempt,
    }


def _adaptive_online_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate adaptive online-policy decision and reward metrics.

    Tracks per-action selection/application counts, overrides, policy
    updates, reward sums, propensity sums, local cost metrics, and
    delayed-rescue credit attribution.
    """
    adaptive_online_mode = "off"
    adaptive_online_policy_algo = ""
    decisions = 0
    selected_action_counts: dict[str, int] = defaultdict(int)
    applied_action_counts: dict[str, int] = defaultdict(int)
    behavior_action_counts: dict[str, int] = defaultdict(int)
    override_attempts = 0
    updates = 0
    reward_sum = 0.0
    reward_count = 0
    reward_values: list[float] = []
    propensity_sum = 0.0
    propensity_count = 0
    local_tokens_total_sum = 0.0
    local_tokens_total_count = 0
    local_wall_sec_sum = 0.0
    local_wall_sec_count = 0
    local_llm_call_count_sum = 0.0
    local_llm_call_count_count = 0
    reward_progress_score_sum = 0.0
    reward_progress_score_count = 0
    progress_reason_counts: dict[str, int] = defaultdict(int)
    delayed_rescue_bonus_sum = 0.0
    delayed_rescue_bonus_count = 0
    credited_rescues_plus_1 = 0
    credited_rescues_plus_2 = 0

    for record in records:
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        mode_raw = attempt_meta_dict.get(
            "adaptive_online_policy_mode",
            patch_meta_dict.get("adaptive_online_policy_mode"),
        )
        if isinstance(mode_raw, str) and mode_raw and adaptive_online_mode == "off":
            adaptive_online_mode = mode_raw
        algo_raw = attempt_meta_dict.get(
            "adaptive_online_policy_algo",
            patch_meta_dict.get("adaptive_online_policy_algo"),
        )
        if isinstance(algo_raw, str) and algo_raw and not adaptive_online_policy_algo:
            adaptive_online_policy_algo = algo_raw

        decision_id = attempt_meta_dict.get(
            "adaptive_online_decision_id",
            patch_meta_dict.get("adaptive_online_decision_id"),
        )
        has_decision = isinstance(decision_id, str) and bool(decision_id.strip())
        reward_scope = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_reward_scope",
        )
        has_local_scope = reward_scope in ADAPTIVE_ONLINE_REWARD_SCOPES
        if has_decision:
            decisions += 1

        selected_action = attempt_meta_dict.get(
            "adaptive_online_action_selected",
            patch_meta_dict.get("adaptive_online_action_selected"),
        )
        if isinstance(selected_action, str) and selected_action in ADAPTIVE_ONLINE_ACTIONS:
            selected_action_counts[selected_action] += 1
        applied_action = attempt_meta_dict.get(
            "adaptive_online_action_applied",
            patch_meta_dict.get("adaptive_online_action_applied"),
        )
        if isinstance(applied_action, str) and applied_action in ADAPTIVE_ONLINE_ACTIONS:
            applied_action_counts[applied_action] += 1
        behavior_action = attempt_meta_dict.get(
            "adaptive_online_behavior_action",
            patch_meta_dict.get("adaptive_online_behavior_action"),
        )
        if isinstance(behavior_action, str) and behavior_action in ADAPTIVE_ONLINE_ACTIONS:
            behavior_action_counts[behavior_action] += 1
        if (
            isinstance(selected_action, str)
            and isinstance(applied_action, str)
            and selected_action
            and applied_action
            and selected_action != "none"
            and applied_action != "none"
            and selected_action != applied_action
        ):
            override_attempts += 1

        updated_raw = attempt_meta_dict.get(
            "adaptive_online_updated",
            patch_meta_dict.get("adaptive_online_updated"),
        )
        if isinstance(updated_raw, bool) and updated_raw:
            updates += 1

        reward_raw = attempt_meta_dict.get(
            "adaptive_online_reward",
            patch_meta_dict.get("adaptive_online_reward"),
        )
        if has_decision and has_local_scope and isinstance(reward_raw, (int, float)):
            reward_value = float(reward_raw)
            reward_sum += reward_value
            reward_count += 1
            reward_values.append(reward_value)

        reward_progress_score_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_reward_progress_score",
        )
        if has_decision and has_local_scope and isinstance(reward_progress_score_raw, (int, float)):
            reward_progress_score_sum += float(reward_progress_score_raw)
            reward_progress_score_count += 1

        progress_reason_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_progress_reason",
        )
        if has_decision and has_local_scope and isinstance(progress_reason_raw, str):
            progress_reason = progress_reason_raw.strip()
            if progress_reason:
                progress_reason_counts[progress_reason] += 1

        propensity_raw = attempt_meta_dict.get(
            "adaptive_online_propensity",
            patch_meta_dict.get("adaptive_online_propensity"),
        )
        if has_decision and isinstance(propensity_raw, (int, float)):
            propensity_sum += float(propensity_raw)
            propensity_count += 1

        local_tokens_total_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_local_tokens_total",
        )
        if has_decision and has_local_scope and isinstance(local_tokens_total_raw, (int, float)):
            local_tokens_total_sum += float(local_tokens_total_raw)
            local_tokens_total_count += 1

        local_wall_sec_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_local_wall_sec",
        )
        if has_decision and has_local_scope and isinstance(local_wall_sec_raw, (int, float)):
            local_wall_sec_sum += float(local_wall_sec_raw)
            local_wall_sec_count += 1

        local_llm_call_count_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_local_llm_call_count",
        )
        if has_decision and has_local_scope and isinstance(local_llm_call_count_raw, (int, float)):
            local_llm_call_count_sum += float(local_llm_call_count_raw)
            local_llm_call_count_count += 1

        delayed_rescue_bonus_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_reward_delayed_rescue_bonus",
        )
        if has_decision and has_local_scope and isinstance(delayed_rescue_bonus_raw, (int, float)):
            delayed_rescue_bonus = float(delayed_rescue_bonus_raw)
            if delayed_rescue_bonus > 0.0:
                delayed_rescue_bonus_sum += delayed_rescue_bonus
                delayed_rescue_bonus_count += 1
                horizon_attempts = _as_int(
                    _metadata_with_fallback(
                        attempt_meta_dict,
                        patch_meta_dict,
                        key="adaptive_online_reward_credit_horizon_attempts",
                    )
                )
                if horizon_attempts == 1:
                    credited_rescues_plus_1 += 1
                elif horizon_attempts == 2:
                    credited_rescues_plus_2 += 1

    return {
        "adaptive_online_mode": adaptive_online_mode,
        "adaptive_online_policy_algo": adaptive_online_policy_algo,
        "decisions": decisions,
        "selected_action_counts": {
            action: int(selected_action_counts.get(action, 0)) for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "applied_action_counts": {
            action: int(applied_action_counts.get(action, 0)) for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "behavior_action_counts": {
            action: int(behavior_action_counts.get(action, 0)) for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "selected_invoke_attempts": int(selected_action_counts.get("invoke_helper", 0)),
        "applied_invoke_attempts": int(applied_action_counts.get("invoke_helper", 0)),
        "override_attempts": override_attempts,
        "updates": updates,
        "reward_sum": reward_sum,
        "reward_count": reward_count,
        "reward_values": reward_values,
        "propensity_sum": propensity_sum,
        "propensity_count": propensity_count,
        "local_tokens_total_sum": local_tokens_total_sum,
        "local_tokens_total_count": local_tokens_total_count,
        "local_wall_sec_sum": local_wall_sec_sum,
        "local_wall_sec_count": local_wall_sec_count,
        "local_llm_call_count_sum": local_llm_call_count_sum,
        "local_llm_call_count_count": local_llm_call_count_count,
        "reward_progress_score_sum": reward_progress_score_sum,
        "reward_progress_score_count": reward_progress_score_count,
        "progress_reason_counts": dict(progress_reason_counts),
        "delayed_rescue_bonus_sum": delayed_rescue_bonus_sum,
        "delayed_rescue_bonus_count": delayed_rescue_bonus_count,
        "credited_rescues_plus_1": credited_rescues_plus_1,
        "credited_rescues_plus_2": credited_rescues_plus_2,
    }


def _adaptive_budget_policy_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate adaptive budget-policy decision and control metrics.

    Tracks budget increase selections, overrides, per-control-knob
    change counts, effective control values, reward/propensity sums,
    and applied multiplier statistics.
    """
    adaptive_budget_policy_mode = "off"
    adaptive_budget_policy_algo = ""
    adaptive_budget_policy_architecture = ""
    decisions = 0
    selected_increase_attempts = 0
    applied_increase_attempts = 0
    selected_time_attempts = 0
    selected_tokens_attempts = 0
    selected_context_attempts = 0
    applied_time_attempts = 0
    applied_tokens_attempts = 0
    applied_context_attempts = 0
    override_attempts = 0
    updates = 0
    reward_sum = 0.0
    reward_count = 0
    reward_values: list[float] = []
    propensity_sum = 0.0
    propensity_count = 0
    family_propensity_sum = 0.0
    family_propensity_count = 0
    request_budget_exceeded_attempts = 0
    applied_multiplier_sum = 0.0
    applied_multiplier_count = 0
    local_tokens_total_sum = 0.0
    local_tokens_total_count = 0
    local_wall_sec_sum = 0.0
    local_wall_sec_count = 0
    local_llm_call_count_sum = 0.0
    local_llm_call_count_count = 0
    reward_progress_score_sum = 0.0
    reward_progress_score_count = 0
    propose_budget_change_attempts = 0
    max_tokens_change_attempts = 0
    helper_allowance_change_attempts = 0
    stage_a_requests_change_attempts = 0
    stage_a_chars_change_attempts = 0
    retry_history_budget_change_attempts = 0
    compaction_allowance_change_attempts = 0
    response_mode_change_attempts = 0
    partial_override_attempts = 0
    effective_helper_max_calls_sum = 0.0
    effective_helper_max_calls_count = 0
    effective_stage_a_max_requests_sum = 0.0
    effective_stage_a_max_requests_count = 0
    effective_stage_a_max_chars_sum = 0.0
    effective_stage_a_max_chars_count = 0
    effective_retry_history_max_chars_sum = 0.0
    effective_retry_history_max_chars_count = 0
    effective_context_compaction_max_calls_sum = 0.0
    effective_context_compaction_max_calls_count = 0
    effective_openai_max_tokens_sum = 0.0
    effective_openai_max_tokens_count = 0

    for record in records:
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        mode_raw = attempt_meta_dict.get(
            "adaptive_budget_policy_mode",
            patch_meta_dict.get("adaptive_budget_policy_mode"),
        )
        if isinstance(mode_raw, str) and mode_raw and adaptive_budget_policy_mode == "off":
            adaptive_budget_policy_mode = mode_raw
        algo_raw = attempt_meta_dict.get(
            "adaptive_budget_policy_algo",
            patch_meta_dict.get("adaptive_budget_policy_algo"),
        )
        if isinstance(algo_raw, str) and algo_raw and not adaptive_budget_policy_algo:
            adaptive_budget_policy_algo = algo_raw
        architecture_raw = attempt_meta_dict.get(
            "adaptive_budget_policy_architecture",
            patch_meta_dict.get("adaptive_budget_policy_architecture"),
        )
        if (
            isinstance(architecture_raw, str)
            and architecture_raw
            and not adaptive_budget_policy_architecture
        ):
            adaptive_budget_policy_architecture = architecture_raw

        decision_id = attempt_meta_dict.get(
            "adaptive_budget_decision_id",
            patch_meta_dict.get("adaptive_budget_decision_id"),
        )
        has_decision = isinstance(decision_id, str) and bool(decision_id.strip())
        reward_scope = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_reward_scope",
        )
        has_local_scope = reward_scope in ADAPTIVE_BUDGET_REWARD_SCOPES
        applied_controls_value = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_applied_control_values",
        )
        applied_controls = (
            applied_controls_value if isinstance(applied_controls_value, dict) else {}
        )
        if has_decision:
            decisions += 1

        selected_action = attempt_meta_dict.get(
            "adaptive_budget_action_selected",
            patch_meta_dict.get("adaptive_budget_action_selected"),
        )
        if selected_action == "increase_budget":
            selected_increase_attempts += 1
        selected_family = attempt_meta_dict.get(
            "adaptive_budget_family_selected",
            patch_meta_dict.get("adaptive_budget_family_selected"),
        )
        if has_decision and selected_family == "time":
            selected_time_attempts += 1
        if has_decision and selected_family == "tokens":
            selected_tokens_attempts += 1
        if has_decision and selected_family == "context":
            selected_context_attempts += 1
        applied_action = attempt_meta_dict.get(
            "adaptive_budget_action_applied",
            patch_meta_dict.get("adaptive_budget_action_applied"),
        )
        if applied_action == "increase_budget":
            applied_increase_attempts += 1
        applied_family = attempt_meta_dict.get(
            "adaptive_budget_family_applied",
            patch_meta_dict.get("adaptive_budget_family_applied"),
        )
        if has_decision and applied_family == "time":
            applied_time_attempts += 1
        if has_decision and applied_family == "tokens":
            applied_tokens_attempts += 1
        if has_decision and applied_family == "context":
            applied_context_attempts += 1
        family_decision_family = ""
        if isinstance(selected_family, str) and selected_family in ADAPTIVE_BUDGET_FAMILIES:
            family_decision_family = selected_family
        elif isinstance(applied_family, str) and applied_family in ADAPTIVE_BUDGET_FAMILIES:
            family_decision_family = applied_family
        if (
            isinstance(selected_action, str)
            and isinstance(applied_action, str)
            and selected_action
            and applied_action
            and selected_action != "none"
            and applied_action != "none"
            and selected_action != applied_action
        ):
            override_attempts += 1

        updated_raw = attempt_meta_dict.get(
            "adaptive_budget_updated",
            patch_meta_dict.get("adaptive_budget_updated"),
        )
        if isinstance(updated_raw, bool) and updated_raw:
            updates += 1

        reward_raw = attempt_meta_dict.get(
            "adaptive_budget_reward",
            patch_meta_dict.get("adaptive_budget_reward"),
        )
        if has_decision and has_local_scope and isinstance(reward_raw, (int, float)):
            reward_value = float(reward_raw)
            reward_sum += reward_value
            reward_count += 1
            reward_values.append(reward_value)

        reward_progress_score_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_reward_progress_score",
        )
        if has_decision and has_local_scope and isinstance(reward_progress_score_raw, (int, float)):
            reward_progress_score_sum += float(reward_progress_score_raw)
            reward_progress_score_count += 1

        propensity_raw = attempt_meta_dict.get(
            "adaptive_budget_propensity",
            patch_meta_dict.get("adaptive_budget_propensity"),
        )
        if has_decision and isinstance(propensity_raw, (int, float)):
            propensity_sum += float(propensity_raw)
            propensity_count += 1
        family_propensity_raw = attempt_meta_dict.get(
            "adaptive_budget_family_propensity",
            patch_meta_dict.get("adaptive_budget_family_propensity"),
        )
        if (
            has_decision
            and family_decision_family
            and isinstance(family_propensity_raw, (int, float))
        ):
            family_propensity_sum += float(family_propensity_raw)
            family_propensity_count += 1

        request_budget_exceeded_raw = attempt_meta_dict.get(
            "adaptive_budget_request_budget_exceeded",
            patch_meta_dict.get("adaptive_budget_request_budget_exceeded"),
        )
        if (
            has_decision
            and isinstance(request_budget_exceeded_raw, bool)
            and request_budget_exceeded_raw
        ):
            request_budget_exceeded_attempts += 1

        applied_multiplier_raw = attempt_meta_dict.get(
            "adaptive_budget_applied_multiplier",
            patch_meta_dict.get("adaptive_budget_applied_multiplier"),
        )
        if has_decision and isinstance(applied_multiplier_raw, (int, float)):
            applied_multiplier_sum += float(applied_multiplier_raw)
            applied_multiplier_count += 1
        default_multiplier_raw = attempt_meta_dict.get(
            "adaptive_budget_default_multiplier",
            patch_meta_dict.get("adaptive_budget_default_multiplier"),
        )

        applied_controls_changed_value = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_applied_controls_changed",
        )
        if isinstance(applied_controls_changed_value, dict):
            applied_controls_changed = {
                str(key): bool(value) for key, value in applied_controls_changed_value.items()
            }
        else:
            legacy_propose_change = bool(
                isinstance(applied_multiplier_raw, (int, float))
                and isinstance(default_multiplier_raw, (int, float))
                and float(applied_multiplier_raw) > float(default_multiplier_raw)
            )
            applied_controls_changed = {
                "propose_budget": legacy_propose_change,
                "max_tokens": False,
                "helper_allowance": False,
                "stage_a_requests": False,
                "stage_a_chars": False,
                "retry_history_budget": False,
                "compaction_allowance": False,
                "response_mode": False,
            }
        if has_decision and bool(applied_controls_changed.get("propose_budget")):
            propose_budget_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("max_tokens")):
            max_tokens_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("helper_allowance")):
            helper_allowance_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("stage_a_requests")):
            stage_a_requests_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("stage_a_chars")):
            stage_a_chars_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("retry_history_budget")):
            retry_history_budget_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("compaction_allowance")):
            compaction_allowance_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("response_mode")):
            response_mode_change_attempts += 1

        partial_override_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_partial_override",
        )
        if has_decision and bool(partial_override_raw):
            partial_override_attempts += 1

        local_tokens_total_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_local_tokens_total",
        )
        if has_decision and has_local_scope and isinstance(local_tokens_total_raw, (int, float)):
            local_tokens_total_sum += float(local_tokens_total_raw)
            local_tokens_total_count += 1

        local_wall_sec_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_local_wall_sec",
        )
        if has_decision and has_local_scope and isinstance(local_wall_sec_raw, (int, float)):
            local_wall_sec_sum += float(local_wall_sec_raw)
            local_wall_sec_count += 1

        local_llm_call_count_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_local_llm_call_count",
        )
        if has_decision and has_local_scope and isinstance(local_llm_call_count_raw, (int, float)):
            local_llm_call_count_sum += float(local_llm_call_count_raw)
            local_llm_call_count_count += 1

        effective_helper_max_calls_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_helper_max_calls",
        )
        if effective_helper_max_calls_raw is None and "helper_max_calls" in applied_controls:
            effective_helper_max_calls_raw = applied_controls.get("helper_max_calls")
        if has_decision and isinstance(effective_helper_max_calls_raw, (int, float)):
            effective_helper_max_calls_sum += float(effective_helper_max_calls_raw)
            effective_helper_max_calls_count += 1

        effective_stage_a_max_requests_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_stage_a_max_requests",
        )
        if (
            effective_stage_a_max_requests_raw is None
            and "stage_a_max_requests" in applied_controls
        ):
            effective_stage_a_max_requests_raw = applied_controls.get("stage_a_max_requests")
        if has_decision and isinstance(effective_stage_a_max_requests_raw, (int, float)):
            effective_stage_a_max_requests_sum += float(effective_stage_a_max_requests_raw)
            effective_stage_a_max_requests_count += 1

        effective_stage_a_max_chars_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_stage_a_max_chars",
        )
        if effective_stage_a_max_chars_raw is None and "stage_a_max_chars" in applied_controls:
            effective_stage_a_max_chars_raw = applied_controls.get("stage_a_max_chars")
        if has_decision and isinstance(effective_stage_a_max_chars_raw, (int, float)):
            effective_stage_a_max_chars_sum += float(effective_stage_a_max_chars_raw)
            effective_stage_a_max_chars_count += 1

        effective_retry_history_max_chars_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_retry_history_max_chars",
        )
        if (
            effective_retry_history_max_chars_raw is None
            and "retry_history_max_chars" in applied_controls
        ):
            effective_retry_history_max_chars_raw = applied_controls.get("retry_history_max_chars")
        if has_decision and isinstance(effective_retry_history_max_chars_raw, (int, float)):
            effective_retry_history_max_chars_sum += float(effective_retry_history_max_chars_raw)
            effective_retry_history_max_chars_count += 1

        effective_context_compaction_max_calls_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_context_compaction_max_calls",
        )
        if (
            effective_context_compaction_max_calls_raw is None
            and "context_compaction_max_calls" in applied_controls
        ):
            effective_context_compaction_max_calls_raw = applied_controls.get(
                "context_compaction_max_calls"
            )
        if has_decision and isinstance(effective_context_compaction_max_calls_raw, (int, float)):
            effective_context_compaction_max_calls_sum += float(
                effective_context_compaction_max_calls_raw
            )
            effective_context_compaction_max_calls_count += 1

        effective_openai_max_tokens_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_openai_max_tokens",
        )
        if effective_openai_max_tokens_raw is None and "openai_max_tokens" in applied_controls:
            effective_openai_max_tokens_raw = applied_controls.get("openai_max_tokens")
        if has_decision and isinstance(effective_openai_max_tokens_raw, (int, float)):
            effective_openai_max_tokens_sum += float(effective_openai_max_tokens_raw)
            effective_openai_max_tokens_count += 1

    return {
        "adaptive_budget_policy_mode": adaptive_budget_policy_mode,
        "adaptive_budget_policy_algo": adaptive_budget_policy_algo,
        "adaptive_budget_policy_architecture": adaptive_budget_policy_architecture,
        "decisions": decisions,
        "selected_increase_attempts": selected_increase_attempts,
        "applied_increase_attempts": applied_increase_attempts,
        "selected_time_attempts": selected_time_attempts,
        "selected_tokens_attempts": selected_tokens_attempts,
        "selected_context_attempts": selected_context_attempts,
        "applied_time_attempts": applied_time_attempts,
        "applied_tokens_attempts": applied_tokens_attempts,
        "applied_context_attempts": applied_context_attempts,
        "override_attempts": override_attempts,
        "updates": updates,
        "reward_sum": reward_sum,
        "reward_count": reward_count,
        "reward_values": reward_values,
        "propensity_sum": propensity_sum,
        "propensity_count": propensity_count,
        "family_propensity_sum": family_propensity_sum,
        "family_propensity_count": family_propensity_count,
        "request_budget_exceeded_attempts": request_budget_exceeded_attempts,
        "applied_multiplier_sum": applied_multiplier_sum,
        "applied_multiplier_count": applied_multiplier_count,
        "local_tokens_total_sum": local_tokens_total_sum,
        "local_tokens_total_count": local_tokens_total_count,
        "local_wall_sec_sum": local_wall_sec_sum,
        "local_wall_sec_count": local_wall_sec_count,
        "local_llm_call_count_sum": local_llm_call_count_sum,
        "local_llm_call_count_count": local_llm_call_count_count,
        "reward_progress_score_sum": reward_progress_score_sum,
        "reward_progress_score_count": reward_progress_score_count,
        "propose_budget_change_attempts": propose_budget_change_attempts,
        "max_tokens_change_attempts": max_tokens_change_attempts,
        "helper_allowance_change_attempts": helper_allowance_change_attempts,
        "stage_a_requests_change_attempts": stage_a_requests_change_attempts,
        "stage_a_chars_change_attempts": stage_a_chars_change_attempts,
        "retry_history_budget_change_attempts": retry_history_budget_change_attempts,
        "compaction_allowance_change_attempts": compaction_allowance_change_attempts,
        "response_mode_change_attempts": response_mode_change_attempts,
        "partial_override_attempts": partial_override_attempts,
        "effective_helper_max_calls_sum": effective_helper_max_calls_sum,
        "effective_helper_max_calls_count": effective_helper_max_calls_count,
        "effective_stage_a_max_requests_sum": effective_stage_a_max_requests_sum,
        "effective_stage_a_max_requests_count": effective_stage_a_max_requests_count,
        "effective_stage_a_max_chars_sum": effective_stage_a_max_chars_sum,
        "effective_stage_a_max_chars_count": effective_stage_a_max_chars_count,
        "effective_retry_history_max_chars_sum": effective_retry_history_max_chars_sum,
        "effective_retry_history_max_chars_count": effective_retry_history_max_chars_count,
        "effective_context_compaction_max_calls_sum": effective_context_compaction_max_calls_sum,
        "effective_context_compaction_max_calls_count": effective_context_compaction_max_calls_count,
        "effective_openai_max_tokens_sum": effective_openai_max_tokens_sum,
        "effective_openai_max_tokens_count": effective_openai_max_tokens_count,
    }


def _adaptive_ope_samples(
    *,
    records: list[dict[str, Any]],
    prefix: str,
    valid_actions: set[str],
) -> list[dict[str, Any]]:
    """Extract off-policy evaluation samples from raw attempt records.

    Each sample contains (action, propensity, reward, features,
    feature_scaling_profile) for use in IPS, SNIPS, and DR estimators, plus the feasible
    action set and any decision-model advice needed by the per-sample targets.

    Shadow-mode rows are corrected at read time: the executed action is the deterministic
    baseline there, so its behavior propensity is 1.0 regardless of what the controller
    logged (older logs recorded the bandit's propensity, which inflates IPS weights).
    """
    samples: list[dict[str, Any]] = []
    for record in records:
        if _as_int(record.get("attempt")) <= 0:
            continue
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        decision_id = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_decision_id",
        )
        if not isinstance(decision_id, str) or not decision_id.strip():
            continue

        reward_scope = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_reward_scope",
        )
        expected_reward_scopes = (
            ADAPTIVE_ONLINE_REWARD_SCOPES
            if prefix == "adaptive_online"
            else ADAPTIVE_BUDGET_REWARD_SCOPES
        )
        if reward_scope not in expected_reward_scopes:
            continue

        reward_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_reward",
        )
        reward = _as_float_or_none(reward_raw)
        if reward is None:
            continue

        behavior_action_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_behavior_action",
        )
        if behavior_action_raw is None:
            behavior_action_raw = _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_action_applied",
            )
        behavior_action = str(behavior_action_raw or "").strip()
        if behavior_action not in valid_actions:
            continue

        behavior_propensity_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_behavior_propensity",
        )
        if behavior_propensity_raw is None:
            behavior_propensity_raw = _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_propensity",
            )
        logged_propensity = _sanitize_propensity(
            behavior_propensity_raw,
            floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
        )
        policy_mode = str(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_policy_mode",
            )
            or ""
        ).strip()
        shadow_mode = policy_mode == ADAPTIVE_OPE_SHADOW_POLICY_MODE
        propensity = 1.0 if shadow_mode else logged_propensity

        features_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_feature_vector",
        )
        feature_values = _coerce_feature_values(features_raw)
        feature_scaling_profile = _feature_scaling_profile_bucket(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_feature_scaling_profile",
            )
        )

        samples.append(
            {
                "action": behavior_action,
                "propensity": propensity,
                "reward": reward,
                "features": feature_values,
                "feature_scaling_profile": feature_scaling_profile,
                "propensity_shadow_corrected": shadow_mode and logged_propensity < 1.0,
                **_ope_sample_policy_context(
                    attempt_meta_dict,
                    patch_meta_dict,
                    prefix=prefix,
                    valid_actions=valid_actions,
                ),
            }
        )
    return samples


def _ope_sample_policy_context(
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
    *,
    prefix: str,
    valid_actions: set[str],
) -> dict[str, Any]:
    """Feasible action set and decision-model advice for one OPE sample.

    Rows written before the decision model existed (and every budget-policy row) simply
    yield an empty advice block, which keeps the decision-model targets out of the summary.
    """
    available_raw = _metadata_with_fallback(
        attempt_meta_dict,
        patch_meta_dict,
        key=f"{prefix}_available_actions",
    )
    available_actions: list[str] = []
    if isinstance(available_raw, (list, tuple)):
        for item in available_raw:
            action = str(item).strip()
            if action in valid_actions and action not in available_actions:
                available_actions.append(action)
    dm_probs_raw = _metadata_with_fallback(
        attempt_meta_dict,
        patch_meta_dict,
        key=f"{prefix}_dm_action_probs",
    )
    return {
        "available_actions": available_actions,
        "dm_status": str(
            _metadata_with_fallback(attempt_meta_dict, patch_meta_dict, key=f"{prefix}_dm_status")
            or ""
        ).strip(),
        "dm_action_probs": _coerce_feature_values(dm_probs_raw),
        "dm_atomic_action": str(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_dm_atomic_action",
            )
            or ""
        ).strip(),
        "dm_traceback_sufficient": _as_float_or_none(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_dm_traceback_sufficient",
            )
        ),
        "dm_missing_information": _coerce_feature_values(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_dm_missing_information",
            )
        ),
    }


def _metadata_with_fallback(
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
    *,
    key: str,
) -> Any:
    """Look up *key* in attempt_metadata, falling back to patch_meta."""
    if key in attempt_meta_dict:
        return attempt_meta_dict.get(key)
    return patch_meta_dict.get(key)


def _as_float_or_none(value: Any) -> float | None:
    """Convert *value* to float, returning None on failure or non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _sanitize_propensity(
    value: Any,
    *,
    floor: float = ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
) -> float:
    """Clamp propensity to ``[floor, 1.0]``, defaulting to 1.0."""
    parsed = _as_float_or_none(value)
    if parsed is None:
        return 1.0
    return min(1.0, max(float(floor), parsed))


def _coerce_feature_values(raw: Any) -> dict[str, float]:
    """Coerce a raw dict to ``{str: float}``, dropping unparseable entries."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        parsed = _as_float_or_none(value)
        if parsed is None:
            continue
        out[str(key)] = float(parsed)
    return out


def _feature_scaling_profile_bucket(raw: Any) -> str:
    """Normalize a feature-scaling profile to a bucket label."""
    profile = str(raw or "").strip()
    return profile if profile else "unknown"


def _action_counts_for_samples(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
) -> dict[str, int]:
    """Count how many OPE samples chose each action."""
    action_counts = dict.fromkeys(sorted(action_space), 0)
    for sample in samples:
        action = str(sample.get("action") or "")
        if action in action_counts:
            action_counts[action] += 1
    return action_counts


def _samples_by_feature_scaling_profile(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Partition OPE samples by their feature-scaling profile."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        profile = _feature_scaling_profile_bucket(sample.get("feature_scaling_profile"))
        buckets.setdefault(profile, []).append(sample)
    return {profile: buckets[profile] for profile in sorted(buckets)}


def _action_rates_for_counts(
    *,
    action_counts: dict[str, int],
    total_samples: int,
) -> dict[str, float | None]:
    """Convert action counts into rates with explicit empty-sample handling."""
    if total_samples <= 0:
        return dict.fromkeys(action_counts, None)
    return {action: (count / float(total_samples)) for action, count in action_counts.items()}


def _shannon_entropy_from_counts(action_counts: dict[str, int]) -> float | None:
    """Return action entropy in nats, or ``None`` when no samples are present."""
    total = sum(max(0, int(count)) for count in action_counts.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in action_counts.values():
        if count <= 0:
            continue
        probability = count / float(total)
        entropy -= probability * math.log(probability)
    return entropy


def _adaptive_ope_summary_from_records(
    *,
    records: list[dict[str, Any]],
    prefix: str,
    action_space: set[str],
    target_policies: dict[str, str],
    min_action_support: int,
    propensity_floor: float,
) -> dict[str, Any]:
    """Build an OPE summary directly from raw attempt records."""
    samples = _adaptive_ope_samples(
        records=records,
        prefix=prefix,
        valid_actions=action_space,
    )
    return _adaptive_ope_summary(
        samples=samples,
        action_space=action_space,
        target_policies=target_policies,
        min_action_support=min_action_support,
        propensity_floor=propensity_floor,
    )


def _adaptive_ope_summary(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
    target_policies: dict[str, str],
    min_action_support: int,
    propensity_floor: float,
) -> dict[str, Any]:
    """Build an off-policy evaluation summary from pre-extracted samples.

    Returns sample counts, action counts, and per-target-policy IPS /
    SNIPS / DR estimates.
    """
    action_counts = _action_counts_for_samples(samples=samples, action_space=action_space)
    support_threshold = max(1, int(min_action_support))
    trimmed_propensity_floor = _sanitize_propensity(
        propensity_floor,
        floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
    )
    raw_propensities = [
        _sanitize_propensity(sample.get("propensity"), floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR)
        for sample in samples
    ]
    used_propensities = [
        _sanitize_propensity(propensity, floor=trimmed_propensity_floor)
        for propensity in raw_propensities
    ]
    feature_scaling_profiles = list(_samples_by_feature_scaling_profile(samples))
    action_rates = _action_rates_for_counts(
        action_counts=action_counts,
        total_samples=len(samples),
    )
    targets: dict[str, Any] = {}
    for target_name, target_action in target_policies.items():
        targets[target_name] = _deterministic_target_ope(
            samples=samples,
            target_action=target_action,
            action_space=action_space,
            action_counts=action_counts,
            min_action_support=support_threshold,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
            propensity_floor=trimmed_propensity_floor,
        )
    targets.update(
        _decision_model_targets(
            samples=samples,
            settings=StochasticTargetOpeSettings(
                action_space=action_space,
                min_action_support=support_threshold,
                ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
                propensity_floor=trimmed_propensity_floor,
            ),
        )
    )
    return {
        "sample_count": len(samples),
        "action_counts": action_counts,
        "action_rates": action_rates,
        "action_entropy": _shannon_entropy_from_counts(action_counts),
        "min_action_support": support_threshold,
        "propensity_floor": trimmed_propensity_floor,
        "min_propensity_raw": min(raw_propensities, default=None),
        "min_propensity_used": min(used_propensities, default=None),
        "trimmed_sample_count": sum(
            1
            for raw_propensity, used_propensity in zip(
                raw_propensities, used_propensities, strict=True
            )
            if used_propensity > raw_propensity
        ),
        "max_importance_weight": (
            max((1.0 / propensity) for propensity in used_propensities)
            if used_propensities
            else None
        ),
        "ridge_alpha": ADAPTIVE_OPE_RIDGE_ALPHA,
        "feature_scaling_profiles": feature_scaling_profiles,
        "mean_reward": _mean([float(sample.get("reward") or 0.0) for sample in samples]),
        "shadow_corrected_samples": sum(
            1 for sample in samples if bool(sample.get("propensity_shadow_corrected"))
        ),
        "decision_model_sample_count": sum(
            1 for sample in samples if _decision_model_advice_ok(sample)
        ),
        "target_order": list(targets),
        "targets": targets,
    }


def _support_warning_for_target(
    *,
    matched_samples: int,
    total_samples: int,
    min_action_support: int,
) -> str:
    """Summarize behavior-policy support risk for a deterministic target."""
    if total_samples <= 0:
        return "no_samples"
    if matched_samples <= 0:
        return "no_behavior_support"
    if matched_samples < max(2, int(min_action_support)):
        return "weak_behavior_support"
    return ""


def _finite_estimator_values(target: dict[str, Any]) -> list[tuple[str, float]]:
    """Collect finite estimator outputs from an OPE target summary."""
    values: list[tuple[str, float]] = []
    for name in ("ips", "snips", "dr"):
        raw_value = target.get(name)
        if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
            values.append((name, float(raw_value)))
    return values


def _estimator_agreement_summary(target: dict[str, Any]) -> dict[str, Any]:
    """Flag estimator disagreement without over-interpreting sparse estimates."""
    values = _finite_estimator_values(target)
    estimator_names = [name for name, _value in values]
    if len(values) < 2:
        return {
            "estimators_available": estimator_names,
            "estimator_agreement": "insufficient_estimators",
            "estimator_agreement_warning": "insufficient_estimators",
        }

    numeric_values = [value for _name, value in values]
    span = max(numeric_values) - min(numeric_values)
    scale = max(max(abs(value) for value in numeric_values), 1.0)
    signs = {1 if value > 0.0 else -1 if value < 0.0 else 0 for value in numeric_values}
    agrees = len(signs) <= 1 and span <= max(
        ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_ABS_TOL, scale * ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_REL_TOL
    )
    return {
        "estimators_available": estimator_names,
        "estimator_agreement": "agree" if agrees else "mixed",
        "estimator_agreement_warning": "" if agrees else "mixed_estimator_signal",
    }


def _fit_ridge_action_models(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Fit per-action ridge regression models predicting reward.

    When a per-action subset has at least ``_RIDGECV_MIN_SAMPLES`` rows,
    ``RidgeCV`` with 5-fold cross-validation selects the best alpha from
    ``ADAPTIVE_OPE_RIDGE_ALPHAS``.  Otherwise the hardcoded *ridge_alpha*
    fallback is used.

    Actions with fewer samples than *min_action_support* are skipped.

    Returns:
        Tuple of (feature_names, models dict mapping action to beta).
    """
    feature_names = sorted(
        {
            str(name)
            for sample in samples
            for name in (
                sample.get("features", {}).keys()
                if isinstance(sample.get("features"), dict)
                else []
            )
        }
    )
    dim = len(feature_names) + 1  # +1 for bias term
    models: dict[str, np.ndarray] = {}
    for action in sorted(action_space):
        action_samples = [sample for sample in samples if str(sample.get("action") or "") == action]
        if len(action_samples) < int(min_action_support):
            continue
        X = np.ones((len(action_samples), dim), dtype=float)
        y = np.zeros((len(action_samples),), dtype=float)
        for row_idx, sample in enumerate(action_samples):
            y[row_idx] = float(sample.get("reward") or 0.0)
            features = sample.get("features")
            feature_dict = features if isinstance(features, dict) else {}
            for col_idx, name in enumerate(feature_names, start=1):
                X[row_idx, col_idx] = float(feature_dict.get(name, 0.0) or 0.0)

        if len(action_samples) >= _RIDGECV_MIN_SAMPLES:
            # Enough data: cross-validate to pick the best alpha.
            cv_folds = min(5, len(action_samples))
            model = RidgeCV(
                alphas=ADAPTIVE_OPE_RIDGE_ALPHAS,
                cv=cv_folds,
                fit_intercept=False,  # bias column already in X
            )
            model.fit(X, y)
            beta = np.asarray(model.coef_, dtype=float)
            logger.info(
                "RidgeCV selected alpha=%.4g for action=%r (%d samples, %d-fold)",
                model.alpha_,
                action,
                len(action_samples),
                cv_folds,
            )
        else:
            # Fallback: closed-form ridge with the hardcoded alpha.
            logger.debug(
                "Fallback to hardcoded alpha=%.4g for action=%r (%d samples < %d)",
                ridge_alpha,
                action,
                len(action_samples),
                _RIDGECV_MIN_SAMPLES,
            )
            gram = X.T @ X
            if ridge_alpha > 0.0:
                gram = gram + (np.eye(dim, dtype=float) * float(ridge_alpha))
            rhs = X.T @ y
            try:
                beta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(gram) @ rhs

        models[action] = beta
    return feature_names, models


def _predict_action_reward(
    *,
    features: dict[str, float],
    feature_names: list[str],
    beta: np.ndarray,
) -> float:
    """Predict reward for a single sample using a fitted ridge model."""
    vector = np.ones((len(feature_names) + 1,), dtype=float)
    for idx, name in enumerate(feature_names, start=1):
        vector[idx] = float(features.get(name, 0.0) or 0.0)
    return float(vector @ beta)


def _deterministic_target_ope(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    action_space: set[str],
    action_counts: dict[str, int],
    min_action_support: int,
    ridge_alpha: float,
    propensity_floor: float,
) -> dict[str, Any]:
    """Compute IPS, SNIPS, and DR estimates for a deterministic target.

    The target policy always selects *target_action*.  Returns a dict
    with IPS, SNIPS, DR, effective sample size, and support info.
    """
    total = len(samples)
    matched = 0
    weighted_reward_sum = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    for sample in samples:
        action = str(sample.get("action") or "")
        if action != target_action:
            continue
        propensity = _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
        reward = float(sample.get("reward") or 0.0)
        weight = 1.0 / propensity
        matched += 1
        weighted_reward_sum += weight * reward
        weight_sum += weight
        weight_sq_sum += weight * weight
    ips = (weighted_reward_sum / float(total)) if (total > 0 and matched > 0) else None
    snips = (weighted_reward_sum / weight_sum) if weight_sum > 0.0 else None
    ess = ((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0.0 else None

    dr, dr_reason = _stratified_doubly_robust_estimate(
        samples=samples,
        target_action=target_action,
        action_space=action_space,
        min_action_support=min_action_support,
        ridge_alpha=ridge_alpha,
        propensity_floor=propensity_floor,
    )

    support_warning = _support_warning_for_target(
        matched_samples=matched,
        total_samples=total,
        min_action_support=min_action_support,
    )
    result = {
        "target_action": target_action,
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "effective_sample_size": ess,
        "matched_samples": matched,
        "total_samples": total,
        "support_rate": ((matched / float(total)) if total > 0 else None),
        "support_warning": support_warning,
        "dr_supported": dr is not None,
        "dr_reason": dr_reason if dr is None else "",
    }
    result.update(_estimator_agreement_summary(result))
    return result


def _stratified_doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
    propensity_floor: float,
) -> tuple[float | None, str]:
    """Stratified doubly-robust OPE across feature-scaling profiles.

    Fits per-action ridge models within each feature-scaling stratum,
    then weight-averages stratum-level DR estimates by sample count.

    Returns:
        Tuple of (DR estimate or None, reason string if unsupported).
    """
    total = len(samples)
    if total <= 0:
        return None, "no_samples"
    if target_action not in action_space:
        return None, "target_action_out_of_space"

    weighted_dr_sum = 0.0
    for bucket_samples in _samples_by_feature_scaling_profile(samples).values():
        action_counts = _action_counts_for_samples(
            samples=bucket_samples,
            action_space=action_space,
        )
        feature_names, models = _fit_ridge_action_models(
            samples=bucket_samples,
            action_space=action_space,
            min_action_support=min_action_support,
            ridge_alpha=ridge_alpha,
        )
        observed_actions = {
            action for action in action_space if int(action_counts.get(action, 0)) > 0
        }
        supported_actions = {
            action
            for action in observed_actions
            if int(action_counts.get(action, 0)) >= min_action_support
        }
        # DR remains stricter than IPS/SNIPS: each feature-scaling bucket needs enough
        # support to fit the target model and at least one alternate behavior model before
        # we aggregate.
        if int(action_counts.get(target_action, 0)) < min_action_support:
            return None, "insufficient_action_support"
        if not any(action != target_action for action in supported_actions):
            return None, "insufficient_action_support"
        if any(
            int(action_counts.get(action, 0)) >= min_action_support and action not in models
            for action in observed_actions
        ):
            return None, "missing_action_model"
        if target_action not in models:
            return None, "missing_target_model"
        bucket_dr = _doubly_robust_estimate(
            samples=bucket_samples,
            target_action=target_action,
            feature_names=feature_names,
            models=models,
            propensity_floor=propensity_floor,
        )
        weighted_dr_sum += bucket_dr * len(bucket_samples)
    return weighted_dr_sum / float(total), ""


def _doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    feature_names: list[str],
    models: dict[str, np.ndarray],
    propensity_floor: float,
) -> float:
    """Compute a single-stratum doubly-robust reward estimate.

    Combines the fitted regression prediction with an IPS correction
    term for samples matching the target action.
    """
    terms: list[float] = []
    for sample in samples:
        action = str(sample.get("action") or "")
        propensity = _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
        reward = float(sample.get("reward") or 0.0)
        features_raw = sample.get("features")
        features = features_raw if isinstance(features_raw, dict) else {}
        target_hat = _predict_action_reward(
            features=features,
            feature_names=feature_names,
            beta=models[target_action],
        )
        behavior_hat = target_hat
        if action in models:
            behavior_hat = _predict_action_reward(
                features=features,
                feature_names=feature_names,
                beta=models[action],
            )
        correction = (reward - behavior_hat) / propensity if action == target_action else 0.0
        terms.append(target_hat + correction)
    return float(_mean(terms) or 0.0)


@dataclass(frozen=True)
class StochasticTargetOpeSettings:
    """Estimator settings shared by the per-sample (stochastic) OPE targets."""

    action_space: set[str]
    min_action_support: int
    ridge_alpha: float
    propensity_floor: float


def _decision_model_advice_ok(sample: Mapping[str, Any]) -> bool:
    """Whether a sample carries usable decision-model advice."""
    return str(sample.get("dm_status") or "") == DECISION_MODEL_ADVICE_STATUS_OK


def _ordered_actions(actions: Iterable[str]) -> list[str]:
    """Order actions by the controller's action order, with unknown actions last."""
    unknown_rank = len(_ONLINE_ACTION_RANK)
    return sorted(
        {str(action) for action in actions},
        key=lambda action: (_ONLINE_ACTION_RANK.get(action, unknown_rank), action),
    )


def _sample_available_actions(
    sample: Mapping[str, Any],
    *,
    action_space: set[str],
) -> list[str]:
    """Feasible actions logged for a sample, falling back to the full action space."""
    raw = sample.get("available_actions")
    if isinstance(raw, (list, tuple)):
        logged = [str(action) for action in raw if str(action) in action_space]
        if logged:
            return _ordered_actions(logged)
    return _ordered_actions(action_space)


def _ope_baseline_action(available_actions: Sequence[str]) -> str:
    """Baseline action used when advice is missing, mirroring the controller's fallback."""
    for action in (ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER):
        if action in available_actions:
            return action
    return str(available_actions[0]) if available_actions else ""


def _one_hot_distribution(action: str, available_actions: Sequence[str]) -> dict[str, float]:
    """Target distribution placing all mass on *action*."""
    return {name: (1.0 if name == action else 0.0) for name in available_actions}


def _derived_atomic_action(sample: Mapping[str, Any], available: Sequence[str]) -> str:
    """Atomic mapping applied to logged aux answers; empty when none were logged.

    Only ``evidence_atomic_v1`` maps in code at decision time; other profiles with aux questions
    log the answers but no mapped action. Clef answers a profile's questions jointly, so aux
    answers logged next to the main question approximate what the atomic profile would answer.
    """
    sufficient = _as_float_or_none(sample.get("dm_traceback_sufficient"))
    raw_missing = sample.get("dm_missing_information")
    missing = _coerce_feature_values(raw_missing)
    if sufficient is None and not missing:
        return ""
    action, _reason = atomic_action_from_answers(
        ParsedAnswers(traceback_sufficient=sufficient, missing_information=missing),
        available_actions=available,
        thresholds=DECISION_MODEL_ATOMIC_THRESHOLDS,
    )
    return action


def _decision_model_target_distribution(
    sample: Mapping[str, Any],
    *,
    kind: str,
    action_space: set[str],
) -> dict[str, float]:
    """Per-sample target distribution for one decision-model target *kind*.

    ``greedy`` is the arg-max over the advice probabilities restricted to the feasible
    actions, ``stochastic`` renormalizes them, and ``atomic`` follows the atomic profile's
    mapped action (derived from the logged aux answers when another profile was used).
    Missing or failed advice falls back to the baseline action.
    """
    available = _sample_available_actions(sample, action_space=action_space)
    if not available:
        return {}
    baseline = _one_hot_distribution(_ope_baseline_action(available), available)
    if not _decision_model_advice_ok(sample):
        return baseline
    if kind == "atomic":
        atomic_action = str(sample.get("dm_atomic_action") or "").strip()
        if not atomic_action:
            atomic_action = _derived_atomic_action(sample, available)
        if atomic_action in available:
            return _one_hot_distribution(atomic_action, available)
        return baseline
    raw_probs = sample.get("dm_action_probs")
    probs = raw_probs if isinstance(raw_probs, dict) else {}
    masses = {
        action: max(0.0, _as_float_or_none(probs.get(action, 0.0)) or 0.0) for action in available
    }
    total = sum(masses.values())
    if total <= 0.0:
        return baseline
    if kind == "greedy":
        best = available[0]
        for action in available[1:]:
            if masses[action] > masses[best]:
                best = action
        return _one_hot_distribution(best, available)
    return {action: (mass / total) for action, mass in masses.items()}


def _decision_model_targets(
    *,
    samples: list[dict[str, Any]],
    settings: StochasticTargetOpeSettings,
) -> dict[str, Any]:
    """Decision-model OPE targets; empty when no sample carries usable advice."""
    if not any(_decision_model_advice_ok(sample) for sample in samples):
        return {}
    targets: dict[str, Any] = {}
    for target_name, kind in DECISION_MODEL_TARGET_KINDS:
        target_distributions = [
            _decision_model_target_distribution(
                sample,
                kind=kind,
                action_space=settings.action_space,
            )
            for sample in samples
        ]
        targets[target_name] = _stochastic_target_ope(
            samples=samples,
            target_distributions=target_distributions,
            settings=settings,
        )
    return targets


def _target_importance_weight(
    sample: Mapping[str, Any],
    distribution: Mapping[str, float],
    *,
    propensity_floor: float,
) -> float:
    """Importance weight ``pi(a_i | x_i) / mu_i`` for one sample."""
    target_probability = _as_float_or_none(distribution.get(str(sample.get("action") or ""), 0.0))
    if target_probability is None or target_probability <= 0.0:
        return 0.0
    behavior_propensity = _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
    return float(target_probability) / behavior_propensity


def _mean_target_action_probabilities(
    target_distributions: list[dict[str, float]],
) -> dict[str, float]:
    """Mean target probability per action across samples (zero where infeasible)."""
    total = len(target_distributions)
    if total <= 0:
        return {}
    sums: dict[str, float] = {}
    for distribution in target_distributions:
        for action, probability in distribution.items():
            sums[action] = sums.get(action, 0.0) + float(probability)
    return {action: (sums[action] / float(total)) for action in _ordered_actions(sums)}


def _stochastic_target_ope(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    settings: StochasticTargetOpeSettings,
) -> dict[str, Any]:
    """IPS / SNIPS / ESS / DR for a target that is a distribution per sample."""
    total = len(samples)
    matched = 0
    weighted_reward_sum = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    for sample, distribution in zip(samples, target_distributions, strict=True):
        weight = _target_importance_weight(
            sample,
            distribution,
            propensity_floor=settings.propensity_floor,
        )
        if weight <= 0.0:
            continue
        matched += 1
        weighted_reward_sum += weight * float(sample.get("reward") or 0.0)
        weight_sum += weight
        weight_sq_sum += weight * weight
    ips = (weighted_reward_sum / float(total)) if (total > 0 and matched > 0) else None
    snips = (weighted_reward_sum / weight_sum) if weight_sum > 0.0 else None
    ess = ((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0.0 else None
    dr, dr_reason = _stratified_stochastic_doubly_robust_estimate(
        samples=samples,
        target_distributions=target_distributions,
        settings=settings,
    )
    result: dict[str, Any] = {
        "target_kind": "stochastic",
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "effective_sample_size": ess,
        "matched_samples": matched,
        "total_samples": total,
        "support_rate": ((matched / float(total)) if total > 0 else None),
        "support_warning": _support_warning_for_target(
            matched_samples=matched,
            total_samples=total,
            min_action_support=settings.min_action_support,
        ),
        "dr_supported": dr is not None,
        "dr_reason": dr_reason if dr is None else "",
        "mean_action_probabilities": _mean_target_action_probabilities(target_distributions),
    }
    result.update(_estimator_agreement_summary(result))
    return result


def _stratified_stochastic_doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    settings: StochasticTargetOpeSettings,
) -> tuple[float | None, str]:
    """Stratified DR for a per-sample target, reusing the deterministic reward models.

    Like the deterministic estimator this stratifies by feature-scaling profile and
    weight-averages the stratum estimates.  DR is unsupported as soon as one action that
    receives target mass in a stratum has no reward model there.
    """
    total = len(samples)
    if total <= 0:
        return None, "no_samples"
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, float]]]] = {}
    for sample, distribution in zip(samples, target_distributions, strict=True):
        profile = _feature_scaling_profile_bucket(sample.get("feature_scaling_profile"))
        buckets.setdefault(profile, []).append((sample, distribution))

    weighted_dr_sum = 0.0
    for profile in sorted(buckets):
        bucket_samples = [sample for sample, _distribution in buckets[profile]]
        bucket_distributions = [distribution for _sample, distribution in buckets[profile]]
        feature_names, models = _fit_ridge_action_models(
            samples=bucket_samples,
            action_space=settings.action_space,
            min_action_support=settings.min_action_support,
            ridge_alpha=settings.ridge_alpha,
        )
        reason = _stochastic_dr_support_reason(
            target_distributions=bucket_distributions,
            models=models,
            action_counts=_action_counts_for_samples(
                samples=bucket_samples,
                action_space=settings.action_space,
            ),
            min_action_support=settings.min_action_support,
        )
        if reason:
            return None, reason
        weighted_dr_sum += len(bucket_samples) * _stochastic_doubly_robust_estimate(
            samples=bucket_samples,
            target_distributions=bucket_distributions,
            feature_names=feature_names,
            models=models,
            propensity_floor=settings.propensity_floor,
        )
    return weighted_dr_sum / float(total), ""


def _stochastic_dr_support_reason(
    *,
    target_distributions: list[dict[str, float]],
    models: dict[str, np.ndarray],
    action_counts: dict[str, int],
    min_action_support: int,
) -> str:
    """Reason why DR is unsupported for one stratum, or ``""`` when it is supported."""
    weighted_actions = {
        action
        for distribution in target_distributions
        for action, probability in distribution.items()
        if float(probability) > 0.0
    }
    for action in _ordered_actions(weighted_actions):
        if action in models:
            continue
        if int(action_counts.get(action, 0)) < int(min_action_support):
            return "insufficient_action_support"
        return "missing_action_model"
    return ""


def _stochastic_doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    feature_names: list[str],
    models: dict[str, np.ndarray],
    propensity_floor: float,
) -> float:
    """Single-stratum DR estimate for a per-sample target distribution."""
    terms: list[float] = []
    for sample, distribution in zip(samples, target_distributions, strict=True):
        features_raw = sample.get("features")
        features = features_raw if isinstance(features_raw, dict) else {}
        direct = 0.0
        for action, probability in distribution.items():
            if float(probability) <= 0.0:
                continue
            direct += float(probability) * _predict_action_reward(
                features=features,
                feature_names=feature_names,
                beta=models[action],
            )
        weight = _target_importance_weight(sample, distribution, propensity_floor=propensity_floor)
        correction = 0.0
        if weight > 0.0:
            behavior_hat = _predict_action_reward(
                features=features,
                feature_names=feature_names,
                beta=models[str(sample.get("action") or "")],
            )
            correction = weight * (float(sample.get("reward") or 0.0) - behavior_hat)
        terms.append(direct + correction)
    return float(_mean(terms) or 0.0)


def _looks_like_test_path(path: str) -> bool:
    """Heuristic check whether a file path looks like a test file."""
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return False
    parts = [part.lower() for part in normalized.split("/") if part]
    if not parts:
        return False
    file_name = parts[-1]
    if any(part in {"tests", "test"} for part in parts):
        return True
    if file_name == "conftest.py":
        return True
    if file_name.startswith("test_"):
        return True
    return bool(file_name.endswith("_test.py"))


def _patch_meta_flag_count(records: list[dict[str, Any]], key: str, value: Any) -> int:
    """Count attempts whose ``patch_meta[key]`` equals *value*."""
    return sum(
        1
        for record in records
        if isinstance(record.get("patch_meta"), dict) and record["patch_meta"].get(key) == value
    )


def _test_touch_attempt_metrics(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count patch attempts and how many touch test files.

    Returns:
        Tuple of (patch_attempts, test_touch_attempts).
    """
    patch_attempts = 0
    test_touch_attempts = 0
    for record in records:
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        paths_raw = patch_meta.get("patch_file_paths")
        paths = [str(item) for item in paths_raw] if isinstance(paths_raw, list) else []
        file_count_raw = patch_meta.get("patch_file_count")
        file_count = int(file_count_raw) if isinstance(file_count_raw, (int, float)) else len(paths)
        patch_digest = str(
            patch_meta.get(CURRENT_PATCH_DIFF_HASH_FIELD)
            or patch_meta.get(LEGACY_PATCH_DIFF_HASH_FIELD)
            or ""
        ).strip()
        has_patch = file_count > 0 or bool(patch_digest)
        if not has_patch:
            continue
        patch_attempts += 1
        touches_tests_raw = patch_meta.get("patch_touches_tests")
        if isinstance(touches_tests_raw, bool):
            touches_tests = touches_tests_raw
        else:
            touches_tests = any(_looks_like_test_path(path) for path in paths)
        if touches_tests:
            test_touch_attempts += 1
    return patch_attempts, test_touch_attempts


def _mean_prompt_chars_by_condition(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Group mean stage-B prompt character counts by condition."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition") or "")
        value = row.get("stage_b_prompt_chars_mean")
        if not condition or not isinstance(value, (int, float)):
            continue
        grouped[condition].append(float(value))
    return {condition: _mean(values) for condition, values in sorted(grouped.items())}


def _attempts_to_success(records: list[dict[str, Any]]) -> int | None:
    """Return the first successful attempt number, or None."""
    for record in records:
        if bool(record.get("success")):
            return _as_int(record.get("attempt"))
    return None


def _dedupe_records_by_attempt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One scored row per attempt number, by the shared rule in ``evals.case_outcome``."""
    return [cast(dict[str, Any], record) for record in scored_attempts(records)]


def _attempts_executed(records: list[dict[str, Any]]) -> int:
    """Return the number of attempt records."""
    return len(records)


def _attempts_budget_from_records(records: list[dict[str, Any]]) -> int:
    """Determine the maximum attempt budget from records.

    Prefers ``max_attempts`` in attempt_metadata; falls back to the
    highest observed attempt number.
    """
    max_budget = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        value = attempt_meta.get("max_attempts")
        if isinstance(value, int) and value > max_budget:
            max_budget = value
    if max_budget > 0:
        return max_budget
    return max((_as_int(record.get("attempt")) for record in records), default=0)


def _llm_calls_total(records: list[dict[str, Any]]) -> int:
    """Sum LLM call counts across all attempt records."""
    total = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        value = attempt_meta.get("llm_call_count")
        if isinstance(value, int) and value > 0:
            total += value
    return total


@dataclass(frozen=True)
class TokenUsage:
    """Token counts summed over the attempts that reported them (STA-01, STA-09)."""

    prompt: float | None
    completion: float | None
    total: float | None
    covered_attempts: int


def _token_usage(records: list[dict[str, Any]]) -> TokenUsage:
    """Sum prompt, completion and total tokens over the attempts that reported them.

    An attempt with ``token_usage_available is False`` is not covered. When no attempt
    is covered the totals are ``None``; they used to be 0, so a baseline that timed out
    without usage looked cheapest and token coverage read 100% (STA-01). Falls back to
    legacy ``patch_meta.usage.total_tokens`` for records without structured counts.
    """
    prompt_total = 0.0
    completion_total = 0.0
    total_total = 0.0
    covered = 0
    found_attempt_meta = False
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        if attempt_meta.get("token_usage_available") is False:
            found_attempt_meta = True
            continue
        prompt_tokens = attempt_meta.get("tokens_prompt")
        completion_tokens = attempt_meta.get("tokens_completion")
        total_tokens = attempt_meta.get("tokens_total")
        if not isinstance(prompt_tokens, (int, float)):
            continue
        if not isinstance(completion_tokens, (int, float)):
            continue
        if not isinstance(total_tokens, (int, float)):
            continue
        found_attempt_meta = True
        covered += 1
        prompt_total += float(prompt_tokens)
        completion_total += float(completion_tokens)
        total_total += float(total_tokens)
    if covered:
        return TokenUsage(prompt_total, completion_total, total_total, covered)
    if found_attempt_meta:
        return TokenUsage(None, None, None, 0)

    # Backward-compatible fallback for older records.
    fallback_total = 0.0
    for record in records:
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        usage = patch_meta.get("usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get("total_tokens")
        if isinstance(value, (int, float)):
            covered += 1
            fallback_total += float(value)
    if not covered:
        return TokenUsage(None, None, None, 0)
    return TokenUsage(None, None, fallback_total, covered)


def _token_usage_breakdown(
    records: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    """Return (prompt, completion, total) from :func:`_token_usage`."""
    usage = _token_usage(records)
    return usage.prompt, usage.completion, usage.total


def _legacy_patch_meta_has_usage(patch_meta: dict[str, Any] | None) -> bool:
    if not isinstance(patch_meta, dict):
        return False
    usage = patch_meta.get("usage")
    if not isinstance(usage, dict):
        return False
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        if isinstance(usage.get(key), (int, float)):
            return True
    return False


def _legacy_patch_meta_reasoning_tokens(
    patch_meta: dict[str, Any] | None,
) -> tuple[float | None, bool]:
    if not isinstance(patch_meta, dict):
        return None, False
    usage = patch_meta.get("usage")
    if not isinstance(usage, dict):
        return None, False
    candidates = [
        usage,
        usage.get("output_tokens_details"),
        usage.get("completion_tokens_details"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("reasoning_tokens")
        if isinstance(value, (int, float)):
            return float(value), True
    return None, False


def _reasoning_token_usage_breakdown(
    records: list[dict[str, Any]],
) -> tuple[float | None, bool, float | None]:
    """Return case-level reasoning tokens with covered-only semantics."""
    total_reasoning = 0.0
    applicable_attempts = 0
    found_attempt_meta = False
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        if isinstance(attempt_meta, dict):
            found_attempt_meta = True
            llm_calls = attempt_meta.get("llm_call_count")
            token_usage_available = attempt_meta.get("token_usage_available")
            applicable = bool(
                (isinstance(llm_calls, int) and llm_calls > 0)
                or token_usage_available is True
                or _legacy_patch_meta_has_usage(
                    patch_meta if isinstance(patch_meta, dict) else None
                )
            )
            if not applicable:
                continue
            applicable_attempts += 1
            if attempt_meta.get("reasoning_token_usage_available") is True:
                value = attempt_meta.get("tokens_reasoning")
                if isinstance(value, (int, float)):
                    total_reasoning += float(value)
                    continue
            legacy_value, legacy_available = _legacy_patch_meta_reasoning_tokens(
                patch_meta if isinstance(patch_meta, dict) else None
            )
            if legacy_available and legacy_value is not None:
                total_reasoning += legacy_value
                continue
            return None, False, None
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else None
        if not _legacy_patch_meta_has_usage(patch_meta_dict):
            continue
        applicable_attempts += 1
        legacy_value, legacy_available = _legacy_patch_meta_reasoning_tokens(patch_meta_dict)
        if legacy_available and legacy_value is not None:
            total_reasoning += legacy_value
            continue
        return None, False, None

    if applicable_attempts > 0:
        return total_reasoning, True, total_reasoning / float(applicable_attempts)
    if found_attempt_meta:
        return None, False, None

    fallback_total = 0.0
    fallback_found = False
    for record in records:
        patch_meta = record.get("patch_meta")
        legacy_value, legacy_available = _legacy_patch_meta_reasoning_tokens(
            patch_meta if isinstance(patch_meta, dict) else None
        )
        if not legacy_available or legacy_value is None:
            continue
        fallback_found = True
        fallback_total += legacy_value
    if not fallback_found:
        return None, False, None
    return fallback_total, True, fallback_total


def _tokens_used(records: list[dict[str, Any]]) -> float | None:
    """Total tokens used across all attempt records."""
    _prompt, _completion, total = _token_usage_breakdown(records)
    return total


def _retry_history_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate retry-history character, turn, and truncation metrics."""
    attempts = 0
    chars_sum = 0
    turns_sum = 0
    budget_applied_attempts = 0
    truncated_attempts = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        attempts += 1
        chars_value = attempt_meta.get("retry_history_chars")
        if isinstance(chars_value, (int, float)):
            chars_sum += int(chars_value)
        turns_value = attempt_meta.get("retry_history_turns_included")
        if isinstance(turns_value, (int, float)):
            turns_sum += int(turns_value)
        if bool(attempt_meta.get("retry_history_budget_applied")):
            budget_applied_attempts += 1
        if bool(attempt_meta.get("retry_history_truncated")):
            truncated_attempts += 1
    return {
        "attempts": attempts,
        "chars_sum": chars_sum,
        "turns_sum": turns_sum,
        "budget_applied_attempts": budget_applied_attempts,
        "truncated_attempts": truncated_attempts,
    }


def _runtime_used(records: list[dict[str, Any]]) -> float | None:
    """Compute total wall-clock runtime from records.

    Tries ``attempt_wall_clock_sec`` first, then
    ``cumulative_runtime_sec``, then falls back to summing legacy
    ``before_duration_sec`` / ``after_duration_sec`` fields.
    """
    if not records:
        return None
    attempt_runtime_values = [
        float(record["attempt_wall_clock_sec"])
        for record in records
        if isinstance(record.get("attempt_wall_clock_sec"), (int, float))
    ]
    if len(attempt_runtime_values) == len(records):
        return sum(attempt_runtime_values)
    cumulative_values = [
        float(record["cumulative_runtime_sec"])
        for record in records
        if isinstance(record.get("cumulative_runtime_sec"), (int, float))
    ]
    if cumulative_values:
        return max(cumulative_values)
    total = 0.0
    found = False
    first_before = records[0].get("before_duration_sec")
    if isinstance(first_before, (int, float)):
        total += float(first_before)
        found = True
    for record in records:
        after = record.get("after_duration_sec")
        if isinstance(after, (int, float)):
            total += float(after)
            found = True
    return total if found else None


def _baseline_metrics_from_record(record: dict[str, Any]) -> dict[str, Any]:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return {}
    baseline_metrics = patch_meta.get("baseline_metrics")
    return baseline_metrics if isinstance(baseline_metrics, dict) else {}


def _baseline_metric_last_bool(records: list[dict[str, Any]], key: str) -> bool | None:
    for record in reversed(records):
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, bool):
            return value
    return None


def _baseline_metric_last_str(records: list[dict[str, Any]], key: str) -> str | None:
    for record in reversed(records):
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _baseline_metric_sum(records: list[dict[str, Any]], key: str) -> float | None:
    total = 0.0
    found = False
    for record in records:
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total += float(value)
            found = True
    return total if found else None


def _baseline_metric_last_int(records: list[dict[str, Any]], key: str) -> int | None:
    for record in reversed(records):
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_external_baseline_condition(condition: str) -> bool:
    return condition_family(condition) == CONDITION_FAMILY_EXTERNAL_BASELINE


_TIMING_BREAKDOWN_KEYS = (
    "stage_a_llm_sec",
    "stage_a_context_fulfillment_sec",
    "stage_a_total_sec",
    "snapshot_read_sec",
    "adaptive_controller_prepare_sec",
    "adaptive_controller_summarizer_sec",
    "adaptive_controller_sidecar_build_sec",
    "adaptive_controller_sidecar_write_sec",
    "baseline_case_conversion_sec",
    "baseline_sidecar_load_sec",
    "context_compaction_llm_sec",
    "context_compaction_total_sec",
    "prompt_assembly_sec",
    "stage_b_llm_sec",
    "baseline_internal_total_sec",
    "baseline_eval_sec",
    "baseline_orchestration_sec",
    "patch_apply_sec",
    "syntax_preflight_sec",
    "test_before_sec",
    "test_after_sec",
    "recovery_llm_sec",
    "recovery_total_sec",
    "llmdebug_total_sec",
    "llm_total_sec",
    "overhead_sec",
)

_TOKEN_BREAKDOWN_KEYS = (
    "stage_a_tokens_prompt",
    "stage_a_tokens_completion",
    "stage_a_tokens_total",
    "stage_b_tokens_prompt",
    "stage_b_tokens_completion",
    "stage_b_tokens_total",
    "compaction_tokens_prompt",
    "compaction_tokens_completion",
    "compaction_tokens_total",
    "recovery_tokens_prompt",
    "recovery_tokens_completion",
    "recovery_tokens_total",
    "tool_response_chars",
    "snapshot_chars",
)


def _timing_breakdown_mean(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute mean of each timing breakdown field across attempt records."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TIMING_BREAKDOWN_KEYS}
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        timing = attempt_meta.get("timing_breakdown")
        if not isinstance(timing, dict):
            continue
        for key in _TIMING_BREAKDOWN_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)):
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _token_breakdown_mean(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute mean of each token breakdown field across attempt records."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TOKEN_BREAKDOWN_KEYS}
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        tokens = attempt_meta.get("token_breakdown")
        if not isinstance(tokens, dict):
            continue
        for key in _TOKEN_BREAKDOWN_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)):
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _timing_breakdown_sum(records: list[dict[str, Any]]) -> dict[str, float]:
    """Sum each timing breakdown field across attempt records."""
    totals: dict[str, float] = dict.fromkeys(_TIMING_BREAKDOWN_KEYS, 0.0)
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        timing = attempt_meta.get("timing_breakdown")
        if not isinstance(timing, dict):
            continue
        for key in _TIMING_BREAKDOWN_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)):
                totals[key] += float(value)
    return totals


def _token_breakdown_sum(records: list[dict[str, Any]]) -> dict[str, int]:
    """Sum each token breakdown field across attempt records."""
    totals: dict[str, int] = dict.fromkeys(_TOKEN_BREAKDOWN_KEYS, 0)
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        tokens = attempt_meta.get("token_breakdown")
        if not isinstance(tokens, dict):
            continue
        for key in _TOKEN_BREAKDOWN_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def _condition_timing_breakdown_mean(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Mean of timing breakdown sums across case-condition rows."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TIMING_BREAKDOWN_KEYS}
    for row in rows:
        timing = row.get("timing_breakdown_sum")
        if not isinstance(timing, dict):
            continue
        for key in _TIMING_BREAKDOWN_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)) and value > 0:
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _condition_token_breakdown_mean(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Mean of token breakdown sums across case-condition rows."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TOKEN_BREAKDOWN_KEYS}
    for row in rows:
        tokens = row.get("token_breakdown_sum")
        if not isinstance(tokens, dict):
            continue
        for key in _TOKEN_BREAKDOWN_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)) and value > 0:
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _uplift_success_rate(summary: dict[str, Any]) -> float | None:
    """Uplift from base_repair to layered_adaptive_llm."""
    return _uplift_between_conditions(
        summary=summary,
        a_condition="base_repair",
        b_condition="layered_adaptive_llm",
    )


def _uplift_between_conditions(
    *,
    summary: dict[str, Any],
    a_condition: str,
    b_condition: str,
) -> float | None:
    """Compute success-rate uplift (b - a) between two conditions."""
    a = summary.get(a_condition)
    b = summary.get(b_condition)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    ar = a.get("success_rate")
    br = b.get("success_rate")
    if not isinstance(ar, (int, float)) or not isinstance(br, (int, float)):
        return None
    return float(br - ar)


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the human-readable evaluation summary to stdout."""
    print("Overall:")
    _print_bucket(summary["overall"])

    print("\nBy category:")
    by_category = summary["by_category"]
    for category, bucket in by_category.items():
        print(f"- {category}:")
        _print_bucket(bucket, prefix="  ")

    print("\nBy difficulty:")
    by_difficulty = summary["by_difficulty"]
    for difficulty, bucket in by_difficulty.items():
        print(f"- {difficulty}:")
        _print_bucket(bucket, prefix="  ")

    print("\nBy snapshot dependency:")
    by_snapshot_dependency = summary["by_snapshot_dependency"]
    for dependency, bucket in by_snapshot_dependency.items():
        print(f"- {dependency}:")
        _print_bucket(bucket, prefix="  ")

    if "by_bug_source" in summary:
        print("\nBy bug source:")
        for source, bucket in summary["by_bug_source"].items():
            print(f"- {source}:")
            _print_bucket(bucket, prefix="  ")

    if "by_contamination_risk" in summary:
        print("\nBy contamination risk:")
        for risk, bucket in summary["by_contamination_risk"].items():
            print(f"- {risk}:")
            _print_bucket(bucket, prefix="  ")

    _print_input_errors(summary.get("input_errors"))


def _print_bucket(bucket: dict[str, Any], prefix: str = "") -> None:
    """Print a single summary bucket with conditions, uplifts, and controls."""
    condition_items = [
        (condition, stats)
        for condition, stats in sorted(bucket.items())
        if isinstance(stats, dict) and "cases" in stats and "success_rate" in stats
    ]
    for condition, stats in condition_items:
        cases = int(stats.get("cases", 0))
        successes = int(stats.get("successes", 0))
        rate = float(stats.get("success_rate", 0.0))
        med_attempts = stats.get("median_attempts_to_success")
        mean_tokens = stats.get("mean_tokens_total", stats.get("mean_token_usage"))
        mean_prompt_tokens = stats.get("mean_tokens_prompt")
        mean_completion_tokens = stats.get("mean_tokens_completion")
        mean_reasoning_tokens = stats.get("mean_tokens_reasoning")
        mean_tokens_per_attempt = stats.get("mean_tokens_per_attempt")
        mean_reasoning_tokens_per_attempt = stats.get("mean_reasoning_tokens_per_attempt")
        mean_llm_calls = stats.get("mean_llm_calls_per_case")
        reasoning_token_covered_cases = int(stats.get("reasoning_token_covered_cases", 0) or 0)
        reasoning_token_coverage_rate = stats.get("reasoning_token_coverage_rate")
        mean_attempt_budget = stats.get("mean_attempts_budget")
        mean_attempt_ratio = stats.get("mean_attempt_ratio_when_successful")
        solve_dist = stats.get("solve_on_attempt_distribution")
        solve_dist_rendered = ""
        if isinstance(solve_dist, dict) and solve_dist:

            def _solve_key(entry: tuple[str, Any]) -> int:
                try:
                    return int(entry[0])
                except (TypeError, ValueError):
                    return 0

            solve_dist_rendered = ", ".join(
                f"{k}:{int(v)}" for k, v in sorted(solve_dist.items(), key=_solve_key)
            )
        mean_runtime = stats.get("mean_runtime_sec")
        patch_attempts = int(stats.get("patch_attempts", 0) or 0)
        test_touch_attempts = int(stats.get("test_touch_attempts", 0) or 0)
        test_touch_attempt_rate = stats.get("test_touch_attempt_rate")
        helper_invoked_cases = int(stats.get("helper_invoked_cases", 0) or 0)
        helper_case_rate = stats.get("helper_invocation_case_rate")
        helper_attempt_rate = stats.get("helper_invocation_attempt_rate")
        helper_materialized_cases = int(stats.get("helper_materialized_cases", 0) or 0)
        helper_materialized_case_rate = stats.get("helper_materialized_case_rate")
        helper_materialized_attempt_rate = stats.get("helper_materialized_attempt_rate")
        helper_low_yield_attempt_rate = stats.get("helper_low_yield_attempt_rate")
        helper_transport_failure_attempt_rate = stats.get("helper_transport_failure_attempt_rate")
        helper_backend_unavailable_attempt_rate = stats.get(
            "helper_backend_unavailable_attempt_rate"
        )
        helper_invalid_payload_attempt_rate = stats.get("helper_invalid_payload_attempt_rate")
        helper_request_budget_exceeded_attempt_rate = stats.get(
            "helper_request_budget_exceeded_attempt_rate"
        )
        helper_effective_success_rate = stats.get("helper_effective_success_rate")
        helper_mean_retries_per_invocation = stats.get("helper_mean_retries_per_invocation")
        helper_mean_transport_retries_per_invocation = stats.get(
            "helper_mean_transport_retries_per_invocation"
        )
        helper_success_when_invoked = stats.get("helper_success_when_invoked")
        helper_rescue_rate = stats.get("helper_rescue_rate")
        mean_helper_calls = stats.get("mean_helper_calls_used_final")
        helper_need_more_true_rate = stats.get("helper_need_more_true_rate")
        helper_budget_exhausted_rate = stats.get("helper_budget_exhausted_rate")
        helper_saturation_guard_rate = stats.get("helper_saturation_guard_rate")
        topup_case_rate = stats.get("adaptive_budget_topup_case_rate")
        topup_attempt_rate = stats.get("adaptive_budget_topup_attempt_rate")
        mean_syntax_cycles = stats.get("mean_syntax_preflight_cycles")
        syntax_fail_attempt_rate = stats.get("syntax_preflight_failed_attempt_rate")
        source_scope_match_rate = stats.get("patch_verifier_source_scope_match_rate")
        adaptive_budget_decisions = int(stats.get("adaptive_budget_decisions", 0) or 0)
        adaptive_budget_selected_rate = stats.get("adaptive_budget_selected_increase_rate")
        adaptive_budget_applied_rate = stats.get("adaptive_budget_applied_increase_rate")
        adaptive_budget_update_rate = stats.get("adaptive_budget_update_rate")
        adaptive_budget_mean_reward = stats.get("adaptive_budget_mean_reward")
        adaptive_budget_mean_multiplier = stats.get("adaptive_budget_mean_applied_multiplier")
        adaptive_budget_budget_exceeded_rate = stats.get(
            "adaptive_budget_request_budget_exceeded_rate"
        )
        adaptive_online_ope = (
            stats.get("adaptive_online_ope")
            if isinstance(stats.get("adaptive_online_ope"), dict)
            else None
        )
        adaptive_budget_ope = (
            stats.get("adaptive_budget_ope")
            if isinstance(stats.get("adaptive_budget_ope"), dict)
            else None
        )
        print(
            f"{prefix}- {condition}: {successes}/{cases} ({rate:.1%}), "
            f"median attempts={_fmt_num(med_attempts)}, "
            f"mean budget={_fmt_num(mean_attempt_budget)}, "
            f"mean success attempt ratio={_fmt_num(mean_attempt_ratio)}, "
            f"mean tokens(total/prompt/completion)={_fmt_num(mean_tokens)}/"
            f"{_fmt_num(mean_prompt_tokens)}/{_fmt_num(mean_completion_tokens)}, "
            f"mean tokens/attempt={_fmt_num(mean_tokens_per_attempt)}, "
            f"token coverage={_fmt_percent(stats.get('token_covered_fraction'))} of attempts, "
            f"mean llm calls={_fmt_num(mean_llm_calls)}, "
            f"mean runtime={_fmt_num(mean_runtime)}s, "
            f"test-touch patches={test_touch_attempts}/{patch_attempts} "
            f"({_fmt_percent(test_touch_attempt_rate)})"
        )
        if reasoning_token_covered_cases > 0:
            print(
                f"{prefix}  reasoning tokens: mean/case={_fmt_num(mean_reasoning_tokens)}, "
                f"mean/covered-attempt={_fmt_num(mean_reasoning_tokens_per_attempt)}, "
                f"coverage={reasoning_token_covered_cases}/{cases} "
                f"({_fmt_percent(reasoning_token_coverage_rate)})"
            )
        if solve_dist_rendered:
            print(f"{prefix}  solve_on_attempt_distribution: {solve_dist_rendered}")
        if helper_invoked_cases > 0 or isinstance(helper_case_rate, (int, float)):
            print(
                f"{prefix}  helper: invoked_cases={helper_invoked_cases}/{cases} "
                f"({_fmt_percent(helper_case_rate)}), "
                f"invocation_attempt_rate={_fmt_percent(helper_attempt_rate)}, "
                f"materialized_cases={helper_materialized_cases}/{cases} "
                f"({_fmt_percent(helper_materialized_case_rate)}), "
                f"materialized_attempt_rate={_fmt_percent(helper_materialized_attempt_rate)}, "
                f"low_yield_attempt_rate={_fmt_percent(helper_low_yield_attempt_rate)}, "
                f"transport_fail_attempt_rate={_fmt_percent(helper_transport_failure_attempt_rate)}, "
                f"backend_unavailable_attempt_rate={_fmt_percent(helper_backend_unavailable_attempt_rate)}, "
                f"invalid_payload_attempt_rate={_fmt_percent(helper_invalid_payload_attempt_rate)}, "
                f"request_budget_exceeded_attempt_rate={_fmt_percent(helper_request_budget_exceeded_attempt_rate)}, "
                f"effective_success_rate={_fmt_percent(helper_effective_success_rate)}, "
                f"success_when_invoked={_fmt_percent(helper_success_when_invoked)}, "
                f"rescue_rate={_fmt_percent(helper_rescue_rate)}, "
                f"mean_calls_used={_fmt_num(mean_helper_calls)}, "
                f"mean_retries_per_invocation={_fmt_num(helper_mean_retries_per_invocation)}, "
                f"mean_transport_retries_per_invocation="
                f"{_fmt_num(helper_mean_transport_retries_per_invocation)}, "
                f"need_more_context_true_rate={_fmt_percent(helper_need_more_true_rate)}, "
                f"budget_exhausted_rate={_fmt_percent(helper_budget_exhausted_rate)}, "
                f"saturation_guard_rate={_fmt_percent(helper_saturation_guard_rate)}"
            )
        if adaptive_budget_decisions > 0:
            print(
                f"{prefix}  budget_policy: decisions={adaptive_budget_decisions}, "
                f"selected_increase_rate={_fmt_percent(adaptive_budget_selected_rate)}, "
                f"applied_increase_rate={_fmt_percent(adaptive_budget_applied_rate)}, "
                f"update_rate={_fmt_percent(adaptive_budget_update_rate)}, "
                f"mean_reward={_fmt_num(adaptive_budget_mean_reward)}, "
                f"mean_multiplier={_fmt_num(adaptive_budget_mean_multiplier)}, "
                f"request_budget_exceeded_rate={_fmt_percent(adaptive_budget_budget_exceeded_rate)}"
            )
        if adaptive_online_ope is not None:
            print(
                f"{prefix}  online_ope: "
                + _format_ope_targets(
                    adaptive_online_ope,
                    ordered_targets=ADAPTIVE_ONLINE_TARGET_POLICY_ORDER,
                )
            )
        if adaptive_budget_ope is not None:
            print(
                f"{prefix}  budget_ope: "
                + _format_ope_targets(
                    adaptive_budget_ope,
                    ordered_targets=ADAPTIVE_BUDGET_TARGET_POLICY_ORDER,
                )
            )
        print(
            f"{prefix}  controls: topup_case_rate={_fmt_percent(topup_case_rate)}, "
            f"topup_attempt_rate={_fmt_percent(topup_attempt_rate)}, "
            f"mean_syntax_cycles={_fmt_num(mean_syntax_cycles)}, "
            f"syntax_failed_attempt_rate={_fmt_percent(syntax_fail_attempt_rate)}, "
            f"source_scope_match_rate={_fmt_percent(source_scope_match_rate)}"
        )
    pairwise_uplifts = bucket.get("pairwise_uplifts")
    if isinstance(pairwise_uplifts, list) and pairwise_uplifts:
        print(f"{prefix}- pairwise_uplifts:")
        for entry in pairwise_uplifts:
            if not isinstance(entry, dict):
                continue
            condition_a = str(entry.get("condition_a") or "")
            condition_b = str(entry.get("condition_b") or "")
            uplift = entry.get("uplift_success_rate")
            if condition_a and condition_b:
                print(f"{prefix}  {condition_a} -> {condition_b}: {_fmt_signed_percent(uplift)}")
    uplift = bucket.get("uplift_success_rate")
    print(f"{prefix}- uplift_success_rate: {_fmt_signed_percent(uplift)}")
    print(
        f"{prefix}- mean_prompt_chars_stage_b: {_fmt_num(bucket.get('mean_prompt_chars_stage_b'))}"
    )
    print(f"{prefix}- mean_retry_history_chars: {_fmt_num(bucket.get('mean_retry_history_chars'))}")
    print(f"{prefix}- mean_retry_history_turns: {_fmt_num(bucket.get('mean_retry_history_turns'))}")
    print(
        f"{prefix}- retry_history_budget_applied_rate: "
        f"{_fmt_percent(bucket.get('retry_history_budget_applied_rate'))}"
    )
    print(
        f"{prefix}- retry_history_truncation_rate: "
        f"{_fmt_percent(bucket.get('retry_history_truncation_rate'))}"
    )
    by_condition = bucket.get("mean_prompt_chars_stage_b_by_condition")
    if isinstance(by_condition, dict) and by_condition:
        rendered = ", ".join(
            f"{condition}={_fmt_num(value)}" for condition, value in sorted(by_condition.items())
        )
        print(f"{prefix}- mean_prompt_chars_stage_b_by_condition: {rendered}")
    print(
        f"{prefix}- stage_a_request_success_rate: "
        f"{_fmt_percent(bucket.get('stage_a_request_success_rate'))}"
    )
    print(f"{prefix}- retry_gate_block_rate: {_fmt_percent(bucket.get('retry_gate_block_rate'))}")
    print(
        f"{prefix}- uplift_staged_vs_monolithic: "
        f"{_fmt_signed_percent(bucket.get('uplift_staged_vs_monolithic'))}"
    )
    print(
        f"{prefix}- uplift_layered_gated_vs_base: "
        f"{_fmt_signed_percent(bucket.get('uplift_layered_gated_vs_base'))}"
    )
    print(
        f"{prefix}- uplift_layered_llm_vs_base: "
        f"{_fmt_signed_percent(bucket.get('uplift_layered_llm_vs_base'))}"
    )
    print(
        f"{prefix}- test_touch_attempts: {int(bucket.get('test_touch_attempts') or 0)}/"
        f"{int(bucket.get('patch_attempts') or 0)} "
        f"({_fmt_percent(bucket.get('test_touch_attempt_rate'))})"
    )


def _print_input_errors(input_errors: Any) -> None:
    """Print input diagnostic counts if any errors were encountered."""
    if not isinstance(input_errors, dict):
        return
    files_unreadable = int(input_errors.get("files_unreadable", 0))
    json_lines_invalid = int(input_errors.get("json_lines_invalid", 0))
    records_skipped = int(input_errors.get("records_skipped", 0))
    duplicate_attempt_rows_seen = int(input_errors.get("duplicate_attempt_rows_seen", 0))
    duplicate_attempt_rows_resolved = int(input_errors.get("duplicate_attempt_rows_resolved", 0))
    if (
        files_unreadable == 0
        and json_lines_invalid == 0
        and records_skipped == 0
        and duplicate_attempt_rows_seen == 0
    ):
        return
    print("\nInput diagnostics:")
    if files_unreadable > 0 or json_lines_invalid > 0 or records_skipped > 0:
        print(
            "- errors: "
            f"files_unreadable={files_unreadable}, "
            f"json_lines_invalid={json_lines_invalid}, "
            f"records_skipped={records_skipped}"
        )
    if duplicate_attempt_rows_seen > 0:
        print(
            "- dedup: "
            f"duplicate_attempt_rows_seen={duplicate_attempt_rows_seen}, "
            f"duplicate_attempt_rows_resolved={duplicate_attempt_rows_resolved}"
        )


def _ope_target_order(ope: dict[str, Any], *, ordered_targets: tuple[str, ...]) -> list[str]:
    """Target names to render, preferring the summary's own ``target_order``.

    Falling back to *ordered_targets* keeps summaries written before ``target_order``
    existed renderable; trailing targets missing from the order (decision-model targets in
    a hand-built summary, for example) are appended in insertion order.
    """
    raw_order = ope.get("target_order")
    order = (
        [str(name) for name in raw_order]
        if isinstance(raw_order, list) and raw_order
        else list(ordered_targets)
    )
    targets_raw = ope.get("targets")
    if isinstance(targets_raw, dict):
        order.extend(str(name) for name in targets_raw if str(name) not in order)
    return order


def _format_ope_targets(ope: dict[str, Any], *, ordered_targets: tuple[str, ...]) -> str:
    """Format OPE target estimates into a one-line summary string."""
    targets_raw = ope.get("targets")
    targets = targets_raw if isinstance(targets_raw, dict) else {}
    parts: list[str] = []
    for target_name in _ope_target_order(ope, ordered_targets=ordered_targets):
        target = targets.get(target_name)
        if not isinstance(target, dict):
            continue
        ips = _fmt_num(target.get("ips"))
        snips = _fmt_num(target.get("snips"))
        dr = _fmt_num(target.get("dr"))
        ess = _fmt_num(target.get("effective_sample_size"))
        matched = int(target.get("matched_samples") or 0)
        total = int(target.get("total_samples") or 0)
        diagnostics: list[str] = []
        support_warning = str(target.get("support_warning") or "")
        if support_warning:
            diagnostics.append(f"support={support_warning}")
        agreement_warning = str(target.get("estimator_agreement_warning") or "")
        if agreement_warning:
            diagnostics.append(f"agreement={agreement_warning}")
        diagnostics_fragment = f" [{', '.join(diagnostics)}]" if diagnostics else ""
        parts.append(
            f"{target_name}: ips={ips}, snips={snips}, dr={dr}, ess={ess}, "
            f"support={matched}/{total}{diagnostics_fragment}"
        )
    return "; ".join(parts) if parts else "n/a"


def _fmt_num(value: Any) -> str:
    """Format a numeric value as a string, or ``'n/a'`` if None."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "n/a"


def _fmt_signed_percent(value: Any) -> str:
    """Format a value as a signed percentage, or ``'n/a'`` if None."""
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+.1%}"


def _fmt_percent(value: Any) -> str:
    """Format a value as a percentage, or ``'n/a'`` if None."""
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.1%}"


def _as_int(value: Any) -> int:
    """Coerce a value to int, returning 0 on failure."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _mean(xs: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty list."""
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _stddev(xs: list[float]) -> float | None:
    """Population standard deviation, or None for an empty list."""
    if not xs:
        return None
    return float(np.std(np.asarray(xs, dtype=float), ddof=0))


def _quantile(xs: list[float], q: float) -> float | None:
    """Quantile of a float list, or None if empty."""
    if not xs:
        return None
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def _ratio(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _median(xs: list[int]) -> float | None:
    """Median of an integer list, or None if empty."""
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return (ys[mid - 1] + ys[mid]) / 2.0


def _expand_paths(items: list[str]) -> list[str]:
    """Expand shell globs in a list of path strings."""
    out: list[str] = []
    for item in items:
        matches = sorted(glob.glob(item))
        out.extend(matches if matches else [item])
    return out


def _run_statistical_analysis(
    rows: list[dict[str, Any]],
    *,
    aggregation: str = "per_run",
    pair_a: str = "base_repair",
    pair_b: str = "layered_adaptive_llm",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run statistical analysis on paired rows and emit pairwise matrix entries.

    Every bootstrap uses *seed*, so identical inputs give identical intervals (STA-07).
    """
    primary = _run_statistical_analysis_pair(
        rows,
        aggregation=aggregation,
        pair_a=pair_a,
        pair_b=pair_b,
        seed=seed,
    )
    conditions = sorted({str(row.get("condition") or "") for row in rows if row.get("condition")})
    pairwise: list[dict[str, Any]] = []
    for idx, condition_a in enumerate(conditions):
        for condition_b in conditions[idx + 1 :]:
            result = _run_statistical_analysis_pair(
                rows,
                aggregation=aggregation,
                pair_a=condition_a,
                pair_b=condition_b,
                seed=seed,
            )
            entry: dict[str, Any] = {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "aggregation": result.get("aggregation", aggregation),
            }
            if "error" in result:
                entry["error"] = result["error"]
            else:
                a_rate = result.get("a_success_rate")
                b_rate = result.get("b_success_rate")
                mcn = result.get("overall_mcnemar", {})
                entry.update(
                    {
                        "n_pairs": int(result.get("n_pairs", 0)),
                        "a_success_rate": a_rate,
                        "b_success_rate": b_rate,
                        "uplift_success_rate": (
                            float(b_rate - a_rate)
                            if isinstance(a_rate, (int, float)) and isinstance(b_rate, (int, float))
                            else None
                        ),
                        "mcnemar_p_value": (
                            float(mcn.get("p_value"))
                            if isinstance(mcn, dict)
                            and isinstance(mcn.get("p_value"), (int, float))
                            else None
                        ),
                        "mcnemar_significant": (
                            bool(mcn.get("significant")) if isinstance(mcn, dict) else None
                        ),
                    }
                )
            pairwise.append(entry)
    primary["pairwise"] = pairwise
    return primary


def _run_statistical_analysis_pair(
    rows: list[dict[str, Any]],
    *,
    aggregation: str,
    pair_a: str,
    pair_b: str,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run statistical analysis for one selected condition pair."""
    from evals.stats import (
        analyze_paired_results,
        any_success_aggregation,
        majority_vote_aggregation,
    )

    if aggregation in ("majority_vote", "any_success"):
        # Group rows by (case_id, condition) across runs.
        from evals.stats import BinaryPair

        by_case: dict[str, list[BinaryPair]] = {}
        case_meta: dict[str, dict[str, str]] = {}
        # First, group by (run_id, case_id) to get matched pairs per run.
        run_case_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            run_id = str(row.get("run_id", ""))
            case_id = str(row.get("case_id", ""))
            condition = str(row.get("condition", ""))
            run_case_map[run_id][case_id][condition] = row
            if case_id not in case_meta:
                case_meta[case_id] = {
                    "category": str(row.get("category", "unknown")),
                    "run_id": run_id,
                }

        # Build per-case lists of (A, B) pairs across runs.
        for cases in run_case_map.values():
            for case_id, conditions in cases.items():
                a = conditions.get(pair_a)
                b = conditions.get(pair_b)
                if a is not None and b is not None:
                    pair: BinaryPair = (bool(a.get("success")), bool(b.get("success")))
                    by_case.setdefault(case_id, []).append(pair)

        if aggregation == "majority_vote":
            aggregated_pairs = majority_vote_aggregation(by_case)
        else:
            aggregated_pairs = any_success_aggregation(by_case)

        # Rebuild rows from aggregated pairs for analyze_paired_results.
        aggregated_rows: list[dict[str, Any]] = []
        for case_id, pair_val in zip(sorted(by_case), aggregated_pairs, strict=True):
            meta = case_meta.get(case_id, {})
            aggregated_rows.append(
                {
                    "run_id": "aggregated",
                    "case_id": case_id,
                    "condition": pair_a,
                    "category": meta.get("category", "unknown"),
                    "success": pair_val[0],
                }
            )
            aggregated_rows.append(
                {
                    "run_id": "aggregated",
                    "case_id": case_id,
                    "condition": pair_b,
                    "category": meta.get("category", "unknown"),
                    "success": pair_val[1],
                }
            )
        result = analyze_paired_results(
            aggregated_rows,
            condition_a=pair_a,
            condition_b=pair_b,
            seed=seed,
        )
        result["aggregation"] = aggregation
        return result

    result = analyze_paired_results(
        rows,
        condition_a=pair_a,
        condition_b=pair_b,
        seed=seed,
    )
    result["aggregation"] = "per_run"
    return result


def _print_stats(stats: dict[str, Any]) -> None:
    """Print statistical analysis results."""
    if "error" in stats:
        print(f"\nStatistics: {stats['error']}")
        return

    print(f"\nStatistical analysis (aggregation={stats.get('aggregation', 'per_run')}):")
    print(
        f"  Pairing: A={stats.get('condition_a', 'base_repair')} "
        f"vs B={stats.get('condition_b', 'layered_adaptive_llm')}"
    )
    print(f"  Matched pairs: {stats.get('n_pairs', 0)}")
    print(f"  A success rate: {stats.get('a_success_rate', 0):.1%}")
    print(f"  B success rate: {stats.get('b_success_rate', 0):.1%}")

    mcn = stats.get("overall_mcnemar", {})
    print(
        f"  McNemar's test: n_01={mcn.get('n_01', 0)}, n_10={mcn.get('n_10', 0)}, "
        f"p={format_p_value(mcn.get('p_value'))} ({mcn.get('method', '?')})"
        f"{' *' if mcn.get('significant') else ''}"
    )

    ch = stats.get("overall_cohens_h", {})
    print(f"  Cohen's h: {ch.get('h', 0):.3f} ({ch.get('magnitude', '?')})")

    bci = stats.get("overall_bootstrap_ci", {})
    print(
        f"  Bootstrap CI ({bci.get('confidence', 0.95):.0%}): "
        f"[{bci.get('ci_lower', 0):.3f}, {bci.get('ci_upper', 0):.3f}]"
    )

    pwr = stats.get("overall_power", {})
    print(f"  Power: {pwr.get('approximate_power', 0):.2f} — {pwr.get('recommendation', '')}")

    per_cat = stats.get("per_category", {})
    if per_cat:
        hb = stats.get("holm_bonferroni", {})
        print("  Per-category:")
        for cat, cat_stats in sorted(per_cat.items()):
            cat_mcn = cat_stats.get("mcnemar", {})
            cat_ch = cat_stats.get("cohens_h", {})
            hb_entry = hb.get(cat, {})
            sig_marker = " *" if hb_entry.get("rejected") else ""
            adj_p = hb_entry.get("adjusted_p")
            adj_str = f", adj_p={format_p_value(adj_p)}" if adj_p is not None else ""
            print(
                f"    {cat}: n={cat_stats.get('n_pairs', 0)}, "
                f"p={format_p_value(cat_mcn.get('p_value'))}{adj_str}, "
                f"h={cat_ch.get('h', 0):.3f}{sig_marker}"
            )

    pairwise = stats.get("pairwise")
    if isinstance(pairwise, list) and pairwise:
        print("  Pairwise matrix:")
        for entry in pairwise:
            if not isinstance(entry, dict):
                continue
            condition_a = str(entry.get("condition_a") or "")
            condition_b = str(entry.get("condition_b") or "")
            if not condition_a or not condition_b:
                continue
            if "error" in entry:
                print(f"    {condition_a} -> {condition_b}: error={entry['error']}")
                continue
            print(
                f"    {condition_a} -> {condition_b}: "
                f"uplift={_fmt_signed_percent(entry.get('uplift_success_rate'))}, "
                f"n={_fmt_num(entry.get('n_pairs'))}, "
                f"p={format_p_value(entry.get('mcnemar_p_value'))}, "
                f"sig={'yes' if entry.get('mcnemar_significant') else 'no'}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
