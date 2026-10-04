"""Bootstrap adaptive policy state JSON from historical event logs.

Reconstructs a warm-start compatible controller state artifact from prior
adaptive helper-policy or budget-policy JSONL events. This is intentionally an
offline utility: it replays logged decisions and observed rewards without
modifying the controller classes themselves.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import _json as json
from . import linucb_core as core
from ._policy_utils import utc_now as _utc_now
from .adaptive_budget_online_policy import (
    BUDGET_POLICY_ACTIONS,
    BUDGET_POLICY_ALGOS,
    BUDGET_POLICY_ARCH_BWK_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_ARCHITECTURES,
    BUDGET_POLICY_FAMILIES,
    BWK_DUAL_STEP,
)
from .adaptive_budget_online_policy import (
    BUDGET_POLICY_ACTIONS as BUDGET_GATE_ACTIONS,
)
from .adaptive_budget_online_policy import (
    STATE_SCHEMA_VERSION as BUDGET_STATE_SCHEMA_VERSION,
)
from .adaptive_online_policy import (
    ONLINE_POLICY_ACTIONS,
    ONLINE_POLICY_ALGOS,
)
from .adaptive_online_policy import (
    STATE_SCHEMA_VERSION as ONLINE_STATE_SCHEMA_VERSION,
)
from .eval_policy_features import CASE_LABEL_FEATURE_NAMES, drop_case_label_features
from .policy_event_log import LoadedPolicyEvent, event_key, load_policy_events
from .policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    is_percentile_feature_scaling_profile,
    normalize_feature_scaling_profile,
)

PolicyKind = Literal["online", "budget"]


@dataclass
class ReplayState(core.LinUCBState):
    """Serializable LinUCB replay state (the shared core state)."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap adaptive online/budget controller state JSON from historical JSONL events."
        )
    )
    parser.add_argument(
        "--events",
        action="append",
        required=True,
        help=(
            "Path or glob to adaptive helper-policy / budget-policy JSONL event files. "
            "Repeatable; events are replayed in chronological order (created_at_utc, run, "
            "then log position) whatever the argument order."
        ),
    )
    parser.add_argument(
        "--output-state",
        required=True,
        help="Output JSON path for the reconstructed controller state artifact.",
    )
    parser.add_argument(
        "--policy-kind",
        choices=["auto", "online", "budget"],
        default="auto",
        help="Controller kind to bootstrap (default: auto-detect).",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1.0,
        help=(
            "L2 prior used for unseen/expanded features during replay. This is not observable "
            "from event logs, so the default matches the shipped templates (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--warmup-decisions",
        type=int,
        default=-1,
        help=(
            "Optional metadata override for the saved state's warmup_decisions field. "
            "Negative values infer a lower-bound from warmup_active telemetry when possible."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Metadata value to write for save_every in the bootstrapped state (default: 50).",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="Fail on first unreadable file or malformed JSON line.",
    )
    parser.add_argument(
        "--keep-case-label-features",
        action="store_true",
        help=(
            "Keep the oracle case-label features "
            f"({', '.join(CASE_LABEL_FEATURE_NAMES)}) when replaying helper-policy events. "
            "They are dropped by default to match the runtime feature mask; pass this flag "
            "only to reproduce historical runs that learned from them."
        ),
    )
    parser.add_argument(
        "--exclude-case-ids",
        default="",
        help=(
            "Case IDs whose events are not replayed: comma-separated, or @file with one ID per "
            "line. Exclude the cases the warm-started run will be evaluated on (POL-15)."
        ),
    )
    parser.add_argument(
        "--exclude-run-ids",
        default="",
        help="Run IDs whose events are not replayed: comma-separated, or @file (POL-15).",
    )
    parser.add_argument(
        "--target-case-ids",
        default="",
        help=(
            "Case IDs of the run this state will warm-start: comma-separated, or @file. "
            "Replayed events on these cases trigger a target_case_overlap warning."
        ),
    )
    args = parser.parse_args(argv)

    expanded = _expand_paths(args.events)
    if not expanded:
        print("No input matched --events.")
        return 1

    try:
        events, input_errors = _load_policy_events(expanded, strict_input=bool(args.strict_input))
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    try:
        excluded_cases = _id_set(args.exclude_case_ids)
        excluded_runs = _id_set(args.exclude_run_ids)
        target_cases = _id_set(args.target_case_ids)
    except OSError as exc:
        print(f"Input error: {exc}")
        return 2
    kept = [
        event
        for event in events
        if str(event.payload.get("case_id") or "") not in excluded_cases
        and str(event.payload.get("run_id") or "") not in excluded_runs
    ]
    excluded_events = len(events) - len(kept)
    events = kept

    if not events:
        print("No valid policy events found.")
        _print_input_errors(input_errors)
        return 1

    overlap = sorted(target_cases & {str(event.payload.get("case_id") or "") for event in events})
    if overlap:
        print(
            f"WARNING: {len(overlap)} replayed case(s) are also target cases "
            f"(e.g. {', '.join(overlap[:3])}); the warm-started policy will have seen the cases "
            "it is evaluated on. Pass --exclude-case-ids to drop them."
        )

    try:
        requested_kind = str(args.policy_kind or "auto").strip().lower()
        policy_kind = _detect_policy_kind(events, requested_kind=requested_kind)
        if policy_kind == "online":
            payload, summary = _bootstrap_online_state(
                events=events,
                l2=max(1e-9, float(args.l2)),
                warmup_decisions=int(args.warmup_decisions),
                save_every=max(1, _safe_int(args.save_every, default=50)),
                source_paths=expanded,
                keep_case_label_features=bool(args.keep_case_label_features),
            )
        else:
            payload, summary = _bootstrap_budget_state(
                events=events,
                l2=max(1e-9, float(args.l2)),
                warmup_decisions=int(args.warmup_decisions),
                save_every=max(1, _safe_int(args.save_every, default=50)),
                source_paths=expanded,
            )
    except ValueError as exc:
        print(f"Bootstrap error: {exc}")
        _print_input_errors(input_errors)
        return 2

    payload["bootstrap_excluded_events"] = excluded_events
    payload["bootstrap_target_case_overlap"] = len(overlap)
    if overlap:
        payload["bootstrap_warning_codes"] = sorted(
            {*payload.get("bootstrap_warning_codes", []), "target_case_overlap"}
        )
        summary["warnings"] = sorted({*summary.get("warnings", []), "target_case_overlap"})
    output_path = Path(str(args.output_state)).expanduser()
    _write_json_atomic(output_path, payload)

    print(
        f"Bootstrapped {policy_kind} policy state from {summary['events_replayed']} events to "
        f"{output_path}"
    )
    if summary.get("warnings"):
        print("Warnings:")
        for warning in list(summary["warnings"]):
            print(f"- {warning}")
    _print_input_errors(input_errors)
    return 0


def _id_set(raw: str) -> set[str]:
    """IDs from a comma-separated string, or from ``@path`` with one ID per line."""
    text = str(raw or "").strip()
    if text.startswith("@"):
        text = Path(text[1:]).expanduser().read_text(encoding="utf-8").replace("\n", ",")
    return {item.strip() for item in text.split(",") if item.strip()}


def _bootstrap_online_state(
    *,
    events: list[LoadedPolicyEvent],
    l2: float,
    warmup_decisions: int,
    save_every: int,
    source_paths: list[str],
    keep_case_label_features: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    algorithm = _infer_uniform_string(
        events,
        field="algorithm",
        default="linucb",
        required=True,
    )
    if algorithm not in ONLINE_POLICY_ALGOS:
        raise ValueError(f"unsupported online algorithm for bootstrap: {algorithm!r}")
    profile = _replayable_profile(events, field="feature_scaling_profile")
    decay = _infer_uniform_float(
        events,
        field="configured_decay",
        default=1.0,
        required=False,
        description="configured_decay",
    )
    alpha = _last_float(events, field="configured_alpha", default=0.8)
    epsilon = _last_float(events, field="configured_epsilon", default=0.05)
    mode = _last_string(events, field="mode", default="on")

    state = ReplayState(action_names=tuple(ONLINE_POLICY_ACTIONS))
    warnings: list[str] = []
    if _events_have_scaling_history_sizes(events, field="feature_scaling_history_sizes"):
        warnings.append("feature_scaling_history_not_reconstructable")

    for loaded in events:
        payload = loaded.payload
        state.decisions += 1
        selected_action = str(payload.get("selected_action") or "")
        if selected_action in ONLINE_POLICY_ACTIONS:
            state.action_counts[selected_action] = (
                int(state.action_counts.get(selected_action, 0)) + 1
            )
        else:
            warnings.append("invalid_selected_action_skipped")
        _count_applied(state, payload.get("applied_action"))
        feature_values = _coerce_feature_mapping(payload.get("feature_values"))
        if not keep_case_label_features:
            masked_features = drop_case_label_features(feature_values)
            if len(masked_features) != len(feature_values):
                warnings.append("case_label_features_dropped")
            feature_values = masked_features
        _ensure_feature_state(
            state=state, cleaned_features=feature_values, l2=l2, warnings=warnings
        )
        _update_feature_scaling_caps(
            state=state,
            raw_caps=payload.get("feature_scaling_caps"),
        )
        if not bool(payload.get("model_updated")):
            continue
        behavior_action = str(payload.get("behavior_action") or "")
        if behavior_action not in ONLINE_POLICY_ACTIONS:
            warnings.append("invalid_behavior_action_skipped")
            continue
        _apply_recency_decay(state=state, action=behavior_action, decay=decay, l2=l2)
        _apply_reward_update(
            state=state,
            action=behavior_action,
            reward=_safe_float(payload.get("reward"), default=0.0),
            feature_values=feature_values,
        )

    payload = {
        "schema_version": ONLINE_STATE_SCHEMA_VERSION,
        "mode": mode,
        "algorithm": algorithm,
        "alpha": alpha,
        "epsilon": epsilon,
        "l2": float(l2),
        "warmup_decisions": _resolved_warmup_decisions(
            events=events,
            requested=warmup_decisions,
            default=200,
        ),
        "save_every": int(save_every),
        "decay": float(decay),
        "feature_scaling_profile": profile,
        **_serialize_replay_state(state),
        "saved_at_utc": _utc_now(),
        "bootstrap_kind": "online",
        "bootstrap_case_label_features_kept": bool(keep_case_label_features),
        "bootstrap_events_replayed": len(events),
        "bootstrap_source_paths": list(source_paths),
        "bootstrap_warning_codes": sorted(set(warnings)),
        "bootstrap_generated_at_utc": _utc_now(),
    }
    return payload, {"events_replayed": len(events), "warnings": sorted(set(warnings))}


def _bootstrap_budget_state(
    *,
    events: list[LoadedPolicyEvent],
    l2: float,
    warmup_decisions: int,
    save_every: int,
    source_paths: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    algorithm = _infer_uniform_string(
        events,
        field="algorithm",
        default="linucb",
        required=True,
    )
    if algorithm not in BUDGET_POLICY_ALGOS:
        raise ValueError(f"unsupported budget algorithm for bootstrap: {algorithm!r}")
    architecture = _infer_uniform_string(
        events,
        field="architecture",
        default=BUDGET_POLICY_ARCH_TIER_BINARY,
        required=False,
    )
    if architecture not in BUDGET_POLICY_ARCHITECTURES:
        raise ValueError(f"unsupported budget architecture for bootstrap: {architecture!r}")
    profile = _replayable_profile(events, field="feature_scaling_profile")
    decay = _infer_uniform_float(
        events,
        field="configured_decay",
        default=1.0,
        required=False,
        description="configured_decay",
    )
    alpha = _last_float(events, field="configured_alpha", default=0.6)
    epsilon = _last_float(events, field="configured_epsilon", default=0.03)
    mode = _last_string(events, field="mode", default="on")

    gate_state = ReplayState(action_names=tuple(BUDGET_GATE_ACTIONS))
    family_state = ReplayState(action_names=tuple(BUDGET_POLICY_FAMILIES))
    warnings: list[str] = []
    if _events_have_scaling_history_sizes(events, field="feature_scaling_history_sizes"):
        warnings.append("feature_scaling_history_not_reconstructable")
    if _events_have_scaling_history_sizes(events, field="family_feature_scaling_history_sizes"):
        warnings.append("family_feature_scaling_history_not_reconstructable")

    bwk_run_token_budget = 0
    bwk_remaining_token_budget = 0
    bwk_cumulative_tokens_charged = 0
    bwk_dual_variable = 0.0
    bwk_cost_sums: dict[str, float] = dict.fromkeys(BUDGET_GATE_ACTIONS, 0.0)
    bwk_cost_counts: dict[str, int] = dict.fromkeys(BUDGET_GATE_ACTIONS, 0)
    if architecture == BUDGET_POLICY_ARCH_BWK_V1:
        bwk_run_token_budget = _infer_uniform_positive_int(
            events,
            field="bwk_run_token_budget",
            description="bwk_run_token_budget",
        )
        bwk_remaining_token_budget = int(bwk_run_token_budget)

    for loaded in events:
        payload = loaded.payload
        gate_state.decisions += 1
        selected_action = str(payload.get("selected_action") or "")
        if selected_action in BUDGET_POLICY_ACTIONS:
            gate_state.action_counts[selected_action] = (
                int(gate_state.action_counts.get(selected_action, 0)) + 1
            )
        else:
            warnings.append("invalid_budget_selected_action_skipped")
        _count_applied(gate_state, payload.get("applied_action"))

        feature_values = _coerce_feature_mapping(payload.get("feature_values"))
        _ensure_feature_state(
            state=gate_state, cleaned_features=feature_values, l2=l2, warnings=warnings
        )
        _update_feature_scaling_caps(
            state=gate_state,
            raw_caps=payload.get("feature_scaling_caps"),
        )

        family_decision_logged = _budget_family_decision_logged(payload)
        family_feature_values = _coerce_feature_mapping(
            payload.get("family_feature_values") or payload.get("feature_values")
        )
        if family_decision_logged:
            family_state.decisions += 1
            selected_family = str(payload.get("selected_family") or "")
            if selected_family in BUDGET_POLICY_FAMILIES:
                family_state.action_counts[selected_family] = (
                    int(family_state.action_counts.get(selected_family, 0)) + 1
                )
            else:
                warnings.append("invalid_budget_selected_family_skipped")
            _count_applied(family_state, payload.get("applied_family"))
            _ensure_feature_state(
                state=family_state,
                cleaned_features=family_feature_values,
                l2=l2,
                warnings=warnings,
            )
            _update_feature_scaling_caps(
                state=family_state,
                raw_caps=payload.get("family_feature_scaling_caps"),
            )

        reward = _safe_float(payload.get("reward"), default=0.0)
        if bool(payload.get("gate_model_updated")):
            behavior_action = str(payload.get("behavior_action") or "")
            if behavior_action in BUDGET_POLICY_ACTIONS:
                _apply_recency_decay(state=gate_state, action=behavior_action, decay=decay, l2=l2)
                _apply_reward_update(
                    state=gate_state,
                    action=behavior_action,
                    reward=reward,
                    feature_values=feature_values,
                )
            else:
                warnings.append("invalid_budget_behavior_action_skipped")

        if bool(payload.get("family_model_updated")):
            behavior_family = str(payload.get("behavior_family") or "")
            if behavior_family in BUDGET_POLICY_FAMILIES:
                _apply_recency_decay(state=family_state, action=behavior_family, decay=decay, l2=l2)
                _apply_reward_update(
                    state=family_state,
                    action=behavior_family,
                    reward=reward,
                    feature_values=family_feature_values,
                )
            else:
                warnings.append("invalid_budget_behavior_family_skipped")

        if architecture == BUDGET_POLICY_ARCH_BWK_V1:
            charged = max(
                0,
                _safe_int(
                    payload.get("bwk_tokens_charged"),
                    default=_safe_int(payload.get("local_tokens_total"), default=0),
                ),
            )
            remaining_before = bwk_remaining_token_budget
            bwk_cumulative_tokens_charged += charged
            bwk_remaining_token_budget = max(0, remaining_before - charged)
            applied_action = str(payload.get("applied_action") or "")
            if applied_action in bwk_cost_counts:
                bwk_cost_sums[applied_action] += float(charged)
                bwk_cost_counts[applied_action] += 1
            if "bwk_target_tokens_per_decision" in payload:
                # Event schema 7+: the pacing dual step (POL-04).
                target = _safe_float(payload.get("bwk_target_tokens_per_decision"), default=0.0)
                if target > 0.0:
                    bwk_dual_variable = max(
                        0.0,
                        float(bwk_dual_variable)
                        + BWK_DUAL_STEP * (float(charged) - target) / target,
                    )
            else:
                # Older logs used the overspend rule; replay it so logged values reconcile.
                denom = max(1, bwk_run_token_budget or charged)
                overspend_tokens = max(0, charged - remaining_before)
                bwk_dual_variable = max(
                    0.0,
                    (float(bwk_dual_variable) * float(decay))
                    + (float(overspend_tokens) / float(denom)),
                )
            after_logged = _safe_int(
                payload.get("bwk_remaining_token_budget_after"),
                default=bwk_remaining_token_budget,
            )
            dual_logged = _safe_float(
                payload.get("bwk_dual_variable_after"),
                default=bwk_dual_variable,
            )
            if after_logged != bwk_remaining_token_budget:
                warnings.append("bwk_remaining_token_budget_mismatch")
            if not np.isclose(dual_logged, bwk_dual_variable, rtol=1e-9, atol=1e-9):
                warnings.append("bwk_dual_variable_mismatch")

    payload = {
        "schema_version": BUDGET_STATE_SCHEMA_VERSION,
        "mode": mode,
        "algorithm": algorithm,
        "architecture": architecture,
        "alpha": alpha,
        "epsilon": epsilon,
        "l2": float(l2),
        "warmup_decisions": _resolved_warmup_decisions(
            events=events,
            requested=warmup_decisions,
            default=20,
        ),
        "save_every": int(save_every),
        "decay": float(decay),
        "feature_scaling_profile": profile,
        "gate_state": _serialize_replay_state(gate_state),
        "family_state": _serialize_replay_state(family_state),
        "bwk_run_token_budget": int(bwk_run_token_budget),
        "bwk_remaining_token_budget": int(bwk_remaining_token_budget),
        "bwk_cumulative_tokens_charged": int(bwk_cumulative_tokens_charged),
        "bwk_dual_variable": float(bwk_dual_variable),
        "bwk_cost_sums": dict(bwk_cost_sums),
        "bwk_cost_counts": dict(bwk_cost_counts),
        "saved_at_utc": _utc_now(),
        "bootstrap_kind": "budget",
        "bootstrap_events_replayed": len(events),
        "bootstrap_source_paths": list(source_paths),
        "bootstrap_warning_codes": sorted(set(warnings)),
        "bootstrap_generated_at_utc": _utc_now(),
    }
    return payload, {"events_replayed": len(events), "warnings": sorted(set(warnings))}


def _expand_paths(items: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        expanded.extend(matches if matches else [str(item)])
    return expanded


def _load_policy_events(
    paths: list[str], *, strict_input: bool
) -> tuple[list[LoadedPolicyEvent], dict[str, int]]:
    """Chronologically ordered, deduplicated events (the shared loader, POL-18)."""
    return load_policy_events(paths, strict_input=strict_input)


def _event_key(payload: dict[str, Any]) -> tuple[str, str, str, int, str] | None:
    return event_key(payload)


def _detect_policy_kind(events: list[LoadedPolicyEvent], *, requested_kind: str) -> PolicyKind:
    seen_kinds = {_payload_policy_kind(item.payload) for item in events}
    if requested_kind in {"online", "budget"}:
        expected_kind = "online" if requested_kind == "online" else "budget"
        if seen_kinds and seen_kinds != {expected_kind}:
            raise ValueError(
                f"requested --policy-kind {expected_kind!r}, but event logs look like "
                f"{sorted(seen_kinds)!r}"
            )
        return expected_kind
    if not seen_kinds:
        raise ValueError("unable to infer policy kind from empty event set")
    if len(seen_kinds) > 1:
        raise ValueError(
            "mixed helper-policy and budget-policy event logs cannot be bootstrapped together"
        )
    kind = next(iter(seen_kinds))
    return "budget" if kind == "budget" else "online"


def _payload_policy_kind(payload: dict[str, Any]) -> PolicyKind:
    if any(
        key in payload
        for key in (
            "architecture",
            "selected_family",
            "applied_family",
            "behavior_family",
            "family_scores",
            "gate_model_updated",
            "family_model_updated",
            "bwk_run_token_budget",
        )
    ):
        return "budget"
    return "online"


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


def _last_string(events: list[LoadedPolicyEvent], *, field: str, default: str) -> str:
    for loaded in reversed(events):
        value = str(loaded.payload.get(field) or "").strip()
        if value:
            return value
    return default


def _last_float(events: list[LoadedPolicyEvent], *, field: str, default: float) -> float:
    for loaded in reversed(events):
        if field in loaded.payload:
            return _safe_float(loaded.payload.get(field), default=default)
    return float(default)


def _infer_uniform_string(
    events: list[LoadedPolicyEvent],
    *,
    field: str,
    default: str,
    required: bool,
) -> str:
    observed = {
        str(item.payload.get(field) or "").strip()
        for item in events
        if str(item.payload.get(field) or "").strip()
    }
    if not observed:
        if required:
            raise ValueError(f"missing required event field {field!r}")
        return default
    if len(observed) > 1:
        raise ValueError(f"mixed {field} values in event logs: {sorted(observed)!r}")
    return next(iter(observed))


def _replayable_profile(events: list[LoadedPolicyEvent], *, field: str) -> str:
    """The scaling profile to bootstrap; ``percentile_v1`` is refused (POL-14).

    Percentile caps are fitted from raw-value history that event logs don't contain, so a
    replayed state would resume in a different feature space from the one its ``A`` and
    ``b`` were built in. Warm-start such runs from their saved state file instead.
    """
    profile = _infer_uniform_profile(events, field=field)
    if is_percentile_feature_scaling_profile(profile):
        raise ValueError(
            f"refusing to bootstrap {profile} state: its scaling caps can't be rebuilt from "
            "event logs; warm-start from the run's saved state file instead"
        )
    return profile


def _infer_uniform_profile(events: list[LoadedPolicyEvent], *, field: str) -> str:
    observed = {
        normalize_feature_scaling_profile(
            str(item.payload.get(field) or DEFAULT_POLICY_FEATURE_SCALING_PROFILE)
        )
        for item in events
    }
    observed.discard("")
    if not observed:
        return DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    if len(observed) > 1:
        raise ValueError(f"mixed {field} values in event logs: {sorted(observed)!r}")
    return next(iter(observed))


def _infer_uniform_float(
    events: list[LoadedPolicyEvent],
    *,
    field: str,
    default: float,
    required: bool,
    description: str,
) -> float:
    values = {
        _safe_float(item.payload.get(field), default=default)
        for item in events
        if field in item.payload
    }
    if not values:
        if required:
            raise ValueError(f"missing required event field {description!r}")
        return float(default)
    if len(values) > 1:
        raise ValueError(f"mixed {description} values in event logs: {sorted(values)!r}")
    return float(next(iter(values)))


def _infer_uniform_positive_int(
    events: list[LoadedPolicyEvent], *, field: str, description: str
) -> int:
    values = {
        max(0, _safe_int(item.payload.get(field), default=0))
        for item in events
        if max(0, _safe_int(item.payload.get(field), default=0)) > 0
    }
    if not values:
        raise ValueError(
            f"missing positive {description} in event logs; pass an event log generated under bwk_v1"
        )
    if len(values) > 1:
        raise ValueError(f"mixed {description} values in event logs: {sorted(values)!r}")
    return int(next(iter(values)))


def _resolved_warmup_decisions(
    *,
    events: list[LoadedPolicyEvent],
    requested: int,
    default: int,
) -> int:
    if int(requested) >= 0:
        return max(0, int(requested))
    warmup_true = [
        max(0, _safe_int(item.payload.get("decision_index"), default=0))
        for item in events
        if bool(item.payload.get("warmup_active"))
    ]
    if not warmup_true:
        return max(0, int(default))
    return max(warmup_true)


def _events_have_scaling_history_sizes(events: list[LoadedPolicyEvent], *, field: str) -> bool:
    for loaded in events:
        raw = loaded.payload.get(field)
        if not isinstance(raw, dict):
            continue
        for value in raw.values():
            if _safe_int(value, default=0) > 0:
                return True
    return False


def _coerce_feature_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    cleaned = {
        str(key): _safe_float(item, default=0.0) for key, item in value.items() if str(key).strip()
    }
    return cleaned


def _ensure_feature_state(
    *,
    state: ReplayState,
    cleaned_features: dict[str, float],
    l2: float,
    warnings: list[str] | None = None,
) -> None:
    """Expand the replay model; flag events that lack known features or reset an arm (POL-21)."""
    if warnings is not None and core.missing_features(state, cleaned_features):
        warnings.append("feature_set_changed_across_events")
    change = core.ensure_feature_state(state, cleaned_features, l2=float(l2))
    if warnings is not None and change.reset_arms:
        warnings.append("arm_reset_to_prior")


def _apply_recency_decay(*, state: ReplayState, action: str, decay: float, l2: float) -> None:
    core.apply_recency_decay(state, action=action, decay=float(decay), l2=float(l2))


def _feature_vector(*, state: ReplayState, feature_values: dict[str, float]) -> np.ndarray:
    return core.feature_vector(
        state, feature_values, to_float=lambda value, _name: _safe_float(value, default=0.0)
    )


def _apply_reward_update(
    *,
    state: ReplayState,
    action: str,
    reward: float,
    feature_values: dict[str, float],
) -> None:
    x = _feature_vector(state=state, feature_values=feature_values)
    core.rank_one_update(state, action=action, reward=reward, x=x)


def _update_feature_scaling_caps(*, state: ReplayState, raw_caps: Any) -> None:
    if not isinstance(raw_caps, dict):
        return
    for key, value in raw_caps.items():
        if not str(key).strip():
            continue
        numeric = _safe_float(value, default=float("nan"))
        if np.isfinite(numeric):
            state.feature_scaling_caps[str(key)] = float(numeric)


def _count_applied(state: ReplayState, raw_action: object) -> None:
    """Count the logged applied action (POL-24); events without one are skipped."""
    action = str(raw_action or "")
    if action in state.applied_action_counts:
        state.applied_action_counts[action] += 1


def _serialize_replay_state(state: ReplayState) -> dict[str, Any]:
    return {
        "feature_names": list(state.feature_names),
        "decisions": int(state.decisions),
        "updates": int(state.updates),
        "action_counts": {
            action: int(state.action_counts.get(action, 0)) for action in state.action_names
        },
        "applied_action_counts": {
            action: int(state.applied_action_counts.get(action, 0)) for action in state.action_names
        },
        "A": {
            action: state.A.get(action, np.empty((0, 0), dtype=float)).tolist()
            for action in state.action_names
        },
        "b": {
            action: state.b.get(action, np.empty((0,), dtype=float)).tolist()
            for action in state.action_names
        },
        "feature_scaling_history": {
            name: [float(item) for item in values]
            for name, values in state.feature_scaling_history.items()
        },
        "feature_scaling_caps": {
            name: float(value) for name, value in state.feature_scaling_caps.items()
        },
    }


def _budget_family_decision_logged(payload: dict[str, Any]) -> bool:
    if str(payload.get("selected_family") or "").strip():
        return True
    if str(payload.get("applied_family") or "").strip():
        return True
    if str(payload.get("family_decision_strategy") or "").strip() not in {"", "not_applicable"}:
        return True
    raw_available = payload.get("available_families")
    if isinstance(raw_available, (list, tuple)):
        return bool(raw_available)
    return False


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _print_input_errors(input_errors: dict[str, int]) -> None:
    for key in (
        "files_unreadable",
        "json_lines_invalid",
        "records_skipped",
        "duplicate_events_seen",
        "duplicate_events_resolved",
    ):
        value = int(input_errors.get(key, 0))
        if value > 0:
            print(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
