"""Export adaptive policy decision/outcome event logs to flat CSV for model training.

Converts the structured JSONL events from adaptive policy runs
(``--adaptive-online-events-path``) into a denormalized CSV suitable for
supervised learning on policy decisions.  Each row is a single decision with
aggregated features, the chosen action, the observed reward, and outcome
signals.

Usage::

    python -m evals.export_policy_dataset \\
        --events evals/results/run1.adaptive_policy.jsonl \\
        --events evals/results/run2.adaptive_policy.jsonl \\
        --output-csv evals/artifacts/policy_training_data.csv

Supports both helper-policy and budget-policy event schema versions via
automatic detection.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Any

from . import _json as json
from .adaptive_online_policy import EVENT_SCHEMA_VERSION as ONLINE_EVENT_SCHEMA_VERSION

POLICY_EVENT_SCHEMA_VERSION = ONLINE_EVENT_SCHEMA_VERSION
# Accepted schema versions are declared per policy kind instead of derived from the controllers'
# current EVENT_SCHEMA_VERSION, so that reading older logs is an explicit decision: helper events
# are read at v4 (pre decision model) and v5 (with the ``decision_model`` block), budget events at
# v4.  ``tests/test_evals_export_policy_dataset.py`` pins the live controller versions against
# these sets, so a controller bump fails loudly here instead of silently dropping v4 logs.
SUPPORTED_ONLINE_EVENT_SCHEMA_VERSIONS = frozenset({4, 5})
SUPPORTED_BUDGET_EVENT_SCHEMA_VERSIONS = frozenset({4})
SUPPORTED_POLICY_EVENT_SCHEMA_VERSIONS = (
    SUPPORTED_ONLINE_EVENT_SCHEMA_VERSIONS | SUPPORTED_BUDGET_EVENT_SCHEMA_VERSIONS
)

_DM_STRING_FIELDS = (
    "role",
    "status",
    "profile",
    "model",
    "top_action",
    "atomic_action",
    "state_sha256",
)
_DM_FLOAT_FIELDS = ("top_prob", "margin", "entropy", "latency_ms")
_DM_OPTIONAL_FLOAT_FIELDS = ("traceback_sufficient", "fix_confidence")
_DM_INT_FIELDS = ("input_tokens",)
DECISION_MODEL_SCALAR_COLUMNS = (
    *(f"dm_{name}" for name in _DM_STRING_FIELDS),
    *(f"dm_{name}" for name in _DM_FLOAT_FIELDS),
    *(f"dm_{name}" for name in _DM_OPTIONAL_FLOAT_FIELDS),
    *(f"dm_{name}" for name in _DM_INT_FIELDS),
    "dm_prior_beta",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export adaptive policy JSONL event logs to a flat CSV dataset "
            "for offline model training."
        )
    )
    parser.add_argument(
        "--events",
        action="append",
        required=True,
        help=(
            "Path or glob to adaptive helper-policy / budget-policy event JSONL files (repeatable)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="evals/results/adaptive_policy_dataset.csv",
        help="Output CSV path (default: evals/results/adaptive_policy_dataset.csv).",
    )
    parser.add_argument(
        "--min-propensity",
        type=float,
        default=1e-6,
        help="Lower-bound for propensity before inverse weighting (default: 1e-6).",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="Fail on first unreadable file or malformed JSON line.",
    )
    args = parser.parse_args(argv)

    min_propensity = max(1e-12, float(args.min_propensity))
    expanded = _expand_paths(args.events)
    if not expanded:
        print("No input matched --events.")
        return 1

    try:
        events, input_errors = _load_events(expanded, strict_input=bool(args.strict_input))
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    if not events:
        print("No valid policy events found.")
        _print_input_errors(input_errors)
        return 1

    rows = _events_to_rows(events=events, min_propensity=min_propensity)
    output_path = Path(str(args.output_csv))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows=rows, output_path=output_path)

    print(f"Exported {len(rows)} policy events to {output_path}")
    _print_input_errors(input_errors)
    return 0


def _expand_paths(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        out.extend(matches if matches else [str(item)])
    return out


def _load_events(
    paths: list[str], *, strict_input: bool
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    input_errors = {
        "files_unreadable": 0,
        "json_lines_invalid": 0,
        "records_skipped": 0,
        "duplicate_events_seen": 0,
        "duplicate_events_resolved": 0,
    }
    deduped: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for src in paths:
        src_path = Path(src)
        try:
            lines = src_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            input_errors["files_unreadable"] += 1
            if strict_input:
                raise ValueError(f"{src}: unable to read ({exc})") from exc
            continue
        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                input_errors["json_lines_invalid"] += 1
                if strict_input:
                    raise ValueError(f"{src}:{line_no}: invalid JSON ({exc})") from exc
                continue
            if not isinstance(payload, dict):
                input_errors["records_skipped"] += 1
                if strict_input:
                    raise ValueError(f"{src}:{line_no}: JSON root is not an object")
                continue
            schema_version = _safe_int(payload.get("schema_version"), default=-1)
            if schema_version not in SUPPORTED_POLICY_EVENT_SCHEMA_VERSIONS:
                input_errors["records_skipped"] += 1
                if strict_input:
                    raise ValueError(
                        f"{src}:{line_no}: unsupported policy event schema_version="
                        f"{schema_version!r}, expected one of "
                        f"{sorted(SUPPORTED_POLICY_EVENT_SCHEMA_VERSIONS)!r}"
                    )
                continue
            key = _event_key(payload)
            if key is None:
                input_errors["records_skipped"] += 1
                if strict_input:
                    raise ValueError(f"{src}:{line_no}: missing run/case/condition/attempt")
                continue
            if key in deduped:
                input_errors["duplicate_events_seen"] += 1
            deduped[key] = payload
    input_errors["duplicate_events_resolved"] = input_errors["duplicate_events_seen"]
    ordered = [deduped[key] for key in sorted(deduped)]
    return ordered, input_errors


def _event_key(payload: dict[str, Any]) -> tuple[str, str, str, int, str] | None:
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


def _events_to_rows(*, events: list[dict[str, Any]], min_propensity: float) -> list[dict[str, Any]]:
    score_action_names = sorted(
        {
            "skip_helper",
            "invoke_helper",
            "keep_budget",
            "increase_budget",
            *(
                str(name)
                for event in events
                for name in (
                    event.get("scores", {}).keys() if isinstance(event.get("scores"), dict) else []
                )
            ),
        }
    )
    family_score_names = sorted(
        {
            "time",
            "tokens",
            "context",
            *(
                str(name)
                for event in events
                for name in (
                    event.get("family_scores", {}).keys()
                    if isinstance(event.get("family_scores"), dict)
                    else []
                )
            ),
        }
    )
    feature_names = sorted(
        {
            str(name)
            for event in events
            for name in (
                event.get("feature_values", {}).keys()
                if isinstance(event.get("feature_values"), dict)
                else []
            )
        }
    )
    family_feature_names = sorted(
        {
            str(name)
            for event in events
            for name in (
                event.get("family_feature_values", {}).keys()
                if isinstance(event.get("family_feature_values"), dict)
                else []
            )
        }
    )
    dm_prob_actions, dm_missing_options = _decision_model_expansions(events)
    rows: list[dict[str, Any]] = []
    for event in events:
        propensity = _safe_float(event.get("propensity"), default=1.0)
        propensity_clipped = max(min_propensity, propensity)
        behavior_propensity = _safe_float(
            event.get("behavior_propensity", event.get("propensity")),
            default=1.0,
        )
        behavior_propensity_clipped = max(min_propensity, behavior_propensity)
        scores = event.get("scores")
        scores_dict = scores if isinstance(scores, dict) else {}
        family_scores = event.get("family_scores")
        family_scores_dict = family_scores if isinstance(family_scores, dict) else {}
        features = event.get("feature_values")
        feature_dict = features if isinstance(features, dict) else {}
        family_features = event.get("family_feature_values")
        family_feature_dict = family_features if isinstance(family_features, dict) else {}
        row: dict[str, Any] = {
            "run_id": str(event.get("run_id") or ""),
            "case_id": str(event.get("case_id") or ""),
            "condition": str(event.get("condition") or ""),
            "attempt": _safe_int(event.get("attempt"), default=0),
            "note": str(event.get("note") or ""),
            "decision_id": str(event.get("decision_id") or ""),
            "decision_index": _safe_int(event.get("decision_index"), default=0),
            "mode": str(event.get("mode") or ""),
            "algorithm": str(event.get("algorithm") or ""),
            "architecture": str(event.get("architecture") or ""),
            "decision_strategy": str(event.get("decision_strategy") or ""),
            "selected_action": str(event.get("selected_action") or ""),
            "applied_action": str(event.get("applied_action") or ""),
            "behavior_action": str(
                event.get("behavior_action") or event.get("applied_action") or ""
            ),
            "default_action": str(event.get("default_action") or ""),
            "selected_family": str(event.get("selected_family") or ""),
            "applied_family": str(event.get("applied_family") or ""),
            "behavior_family": str(event.get("behavior_family") or ""),
            "family_decision_strategy": str(event.get("family_decision_strategy") or ""),
            "override_reason": str(event.get("override_reason") or ""),
            "feature_scaling_profile": str(event.get("feature_scaling_profile") or ""),
            "propensity": propensity,
            "propensity_clipped": propensity_clipped,
            "importance_weight": (1.0 / propensity_clipped),
            "behavior_propensity": behavior_propensity,
            "behavior_propensity_clipped": behavior_propensity_clipped,
            "behavior_importance_weight": (1.0 / behavior_propensity_clipped),
            "family_propensity": _safe_float(event.get("family_propensity"), default=1.0),
            "reward_scope": str(event.get("reward_scope") or ""),
            "reward": _safe_float(event.get("reward"), default=0.0),
            "reward_success_hint": _safe_float(event.get("reward_success_hint"), default=0.0),
            "reward_progress_hint": _safe_float(event.get("reward_progress_hint"), default=0.0),
            "reward_token_penalty": _safe_float(event.get("reward_token_penalty"), default=0.0),
            "reward_latency_penalty": _safe_float(event.get("reward_latency_penalty"), default=0.0),
            "reward_stagnation_penalty": _safe_float(
                event.get("reward_stagnation_penalty"),
                default=0.0,
            ),
            "reward_helper_materialization_bonus": _safe_float(
                event.get("reward_helper_materialization_bonus"), default=0.0
            ),
            "reward_budget_exceeded_penalty": _safe_float(
                event.get("reward_budget_exceeded_penalty"), default=0.0
            ),
            "request_budget_exceeded": 1 if bool(event.get("request_budget_exceeded")) else 0,
            "success": 1 if bool(event.get("success")) else 0,
            "local_llm_call_count": _safe_int(event.get("local_llm_call_count"), default=0),
            "local_tokens_prompt": _safe_int(event.get("local_tokens_prompt"), default=0),
            "local_tokens_completion": _safe_int(event.get("local_tokens_completion"), default=0),
            "local_tokens_total": _safe_int(event.get("local_tokens_total"), default=0),
            "local_wall_sec": _safe_float(event.get("local_wall_sec"), default=0.0),
            "selected_multiplier": _safe_float(event.get("selected_multiplier"), default=0.0),
            "applied_multiplier": _safe_float(event.get("applied_multiplier"), default=0.0),
            "default_multiplier": _safe_float(event.get("default_multiplier"), default=0.0),
            "tier_index_before": _safe_int(event.get("tier_index_before"), default=0),
            "tier_index_after": _safe_int(event.get("tier_index_after"), default=0),
            "gate_model_updated": 1 if bool(event.get("gate_model_updated")) else 0,
            "family_model_updated": 1 if bool(event.get("family_model_updated")) else 0,
            "model_updated": 1 if bool(event.get("model_updated")) else 0,
            "update_reason": str(event.get("update_reason") or ""),
            "created_at_utc": str(event.get("created_at_utc") or ""),
        }
        for action_name in score_action_names:
            row[f"score_{action_name}"] = _safe_float(scores_dict.get(action_name), default=0.0)
        for family_name in family_score_names:
            row[f"family_score_{family_name}"] = _safe_float(
                family_scores_dict.get(family_name),
                default=0.0,
            )
        for feature_name in feature_names:
            row[f"feature_{feature_name}"] = _safe_float(
                feature_dict.get(feature_name), default=0.0
            )
        for feature_name in family_feature_names:
            row[f"family_feature_{feature_name}"] = _safe_float(
                family_feature_dict.get(feature_name),
                default=0.0,
            )
        row.update(
            _decision_model_row_fields(
                event,
                prob_actions=dm_prob_actions,
                missing_options=dm_missing_options,
            )
        )
        rows.append(row)
    return rows


def _decision_model_block(event: dict[str, Any]) -> dict[str, Any]:
    """Return the event's ``decision_model`` block (empty for v4 / no-decision-model events)."""
    block = event.get("decision_model")
    return block if isinstance(block, dict) else {}


def _decision_model_expansions(events: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Union of decision-model action-probability and missing-information option names."""
    prob_actions: set[str] = set()
    missing_options: set[str] = set()
    for event in events:
        block = _decision_model_block(event)
        action_probs = block.get("action_probs")
        if isinstance(action_probs, dict):
            prob_actions.update(str(name) for name in action_probs)
        missing_information = block.get("missing_information")
        if isinstance(missing_information, dict):
            missing_options.update(str(name) for name in missing_information)
    return sorted(prob_actions), sorted(missing_options)


def _decision_model_row_fields(
    event: dict[str, Any],
    *,
    prob_actions: list[str],
    missing_options: list[str],
) -> dict[str, Any]:
    """Flatten one event's ``decision_model`` block into CSV columns.

    Events without the block (schema v4, or runs with the decision model off) get empty
    values everywhere so that "no advice" stays distinguishable from a zero probability.
    """
    dynamic_columns = [
        *(f"dm_p_{action}" for action in prob_actions),
        *(f"dm_missing_{option}" for option in missing_options),
    ]
    block = _decision_model_block(event)
    if not block:
        return dict.fromkeys((*DECISION_MODEL_SCALAR_COLUMNS, *dynamic_columns), "")
    effect = block.get("effect")
    effect_dict = effect if isinstance(effect, dict) else {}
    action_probs = block.get("action_probs")
    action_probs_dict = action_probs if isinstance(action_probs, dict) else {}
    missing_information = block.get("missing_information")
    missing_dict = missing_information if isinstance(missing_information, dict) else {}
    row: dict[str, Any] = {f"dm_{name}": str(block.get(name) or "") for name in _DM_STRING_FIELDS}
    for name in _DM_FLOAT_FIELDS:
        row[f"dm_{name}"] = _safe_float(block.get(name), default=0.0)
    for name in _DM_OPTIONAL_FLOAT_FIELDS:
        row[f"dm_{name}"] = _optional_float_cell(block.get(name))
    for name in _DM_INT_FIELDS:
        row[f"dm_{name}"] = _safe_int(block.get(name), default=0)
    row["dm_prior_beta"] = _optional_float_cell(effect_dict.get("prior_beta"))
    for action in prob_actions:
        row[f"dm_p_{action}"] = _safe_float(action_probs_dict.get(action), default=0.0)
    for option in missing_options:
        row[f"dm_missing_{option}"] = _safe_float(missing_dict.get(option), default=0.0)
    return row


def _optional_float_cell(value: Any) -> float | str:
    """Render an optional numeric field: the float value, or ``""`` when absent."""
    if value is None:
        return ""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    if parsed != parsed:  # NaN check
        return ""
    return parsed


def _write_rows(*, rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_input_errors(input_errors: dict[str, int]) -> None:
    files_unreadable = int(input_errors.get("files_unreadable", 0))
    json_lines_invalid = int(input_errors.get("json_lines_invalid", 0))
    records_skipped = int(input_errors.get("records_skipped", 0))
    duplicate_events_seen = int(input_errors.get("duplicate_events_seen", 0))
    duplicate_events_resolved = int(input_errors.get("duplicate_events_resolved", 0))
    if (
        files_unreadable == 0
        and json_lines_invalid == 0
        and records_skipped == 0
        and duplicate_events_seen == 0
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
    if duplicate_events_seen > 0:
        print(
            "- dedup: "
            f"duplicate_events_seen={duplicate_events_seen}, "
            f"duplicate_events_resolved={duplicate_events_resolved}"
        )


def _safe_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:  # NaN check
        return float(default)
    return out


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


if __name__ == "__main__":
    raise SystemExit(main())
