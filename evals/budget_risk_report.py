"""Summarize budget-cap risk signals from eval JSONL artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import _json as json

_GUARD_STOP_REASONS = frozenset(
    {
        "iteration_budget_exhausted",
        "timeout_exceeded",
        "token_limit_exceeded",
        "error_threshold_exceeded",
        "budget_exceeded",
        "cancelled",
    }
)
_NATIVE_ATTEMPT_FINISH_REASON_CONDITIONS = frozenset(
    {
        "base_repair",
        "cot_repair",
        "layered_adaptive_gated",
        "layered_adaptive_llm",
    }
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object rows")
        rows.append(payload)
    return rows


def _is_success(row: dict[str, Any]) -> bool:
    return bool(row.get("success"))


def _patch_meta(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("patch_meta")
    return payload if isinstance(payload, dict) else {}


def _baseline_meta(row: dict[str, Any]) -> dict[str, Any]:
    payload = _patch_meta(row).get("baseline_meta")
    return payload if isinstance(payload, dict) else {}


def _condition(row: dict[str, Any]) -> str:
    return str(row.get("condition") or "").strip()


def _attempt_finish_reason(row: dict[str, Any]) -> str:
    return str(_patch_meta(row).get("finish_reason") or "").strip()


def _planner_finish_reason(row: dict[str, Any]) -> str:
    patch_meta = _patch_meta(row)
    for key in ("retry_plan_then_patch_meta", "plan_patch_meta"):
        payload = patch_meta.get(key)
        if isinstance(payload, dict):
            value = str(payload.get("finish_reason") or "").strip()
            if value:
                return value
    return ""


def _expects_attempt_finish_reason(row: dict[str, Any]) -> bool:
    patch_meta = _patch_meta(row)
    return (
        _condition(row) in _NATIVE_ATTEMPT_FINISH_REASON_CONDITIONS or "finish_reason" in patch_meta
    )


def _expects_planner_finish_reason(row: dict[str, Any]) -> bool:
    patch_meta = _patch_meta(row)
    if _condition(row) == "cot_repair":
        return True
    if bool(patch_meta.get("prompt_retry_plan_then_patch")):
        return True
    return "plan_patch_meta" in patch_meta or "retry_plan_then_patch_meta" in patch_meta


def _row_has_compaction_trigger(row: dict[str, Any]) -> bool:
    if bool(row.get("context_compaction_auto_overflow_triggered")):
        return True
    return bool(_patch_meta(row).get("context_compaction_auto_overflow_triggered"))


def _prompt_budget_overflow_chars(row: dict[str, Any]) -> int:
    raw_value = row.get("prompt_budget_overflow_chars")
    if raw_value is None:
        raw_value = _patch_meta(row).get("prompt_budget_overflow_chars")
    return int(raw_value or 0)


def _stage_finish_reasons(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for attempt in _baseline_attempts(row):
            value = str(attempt.get(key) or "").strip()
            if value:
                counts[value] += 1
    return counts


def _baseline_attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = _baseline_meta(row).get("attempts")
    if not isinstance(attempts, list):
        return []
    return [item for item in attempts if isinstance(item, dict)]


def _count_stage_unknown(rows: list[dict[str, Any]], key: str) -> int:
    missing = 0
    for row in rows:
        if _condition(row) != "intervenor":
            continue
        for attempt in _baseline_attempts(row):
            value = str(attempt.get(key) or "").strip()
            if not value:
                missing += 1
    return missing


def _baseline_attempt_stop_reasons(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for attempt in _baseline_attempts(row):
            value = str(attempt.get("stop_reason") or "").strip()
            if value:
                counts[value] += 1
    return counts


def _baseline_call_finish_reason_stats(
    rows: list[dict[str, Any]],
) -> tuple[Counter[str], int, int, int]:
    counts: Counter[str] = Counter()
    missing_attempts = 0
    length_finish_total = 0
    compat_recovery_total = 0
    for row in rows:
        for attempt in _baseline_attempts(row):
            length_finish_total += int(attempt.get("call_length_finish_count") or 0)
            compat_recovery_total += int(attempt.get("compat_recovery_count") or 0)
            raw_counts = attempt.get("call_finish_reason_counts")
            if isinstance(raw_counts, dict) and raw_counts:
                for key, value in raw_counts.items():
                    if isinstance(key, str) and isinstance(value, int) and value > 0:
                        counts[key] += value
                continue
            call_records = attempt.get("call_records")
            if isinstance(call_records, list):
                found = False
                for record in call_records:
                    if not isinstance(record, dict):
                        continue
                    finish_reason = str(record.get("finish_reason") or "").strip()
                    if finish_reason:
                        counts[finish_reason] += 1
                        found = True
                if found:
                    continue
            if int(attempt.get("llm_call_count") or 0) > 0:
                missing_attempts += 1
    return counts, missing_attempts, length_finish_total, compat_recovery_total


def _lambda_attempt_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    planned_decomposition_attempts = 0
    executed_decomposition_attempts = 0
    task_detection_call_total = 0
    repl_llm_query_call_total = 0
    unknown_attempts = 0
    for row in rows:
        if str(row.get("condition") or "").strip() != "lambda_rlm":
            continue
        for attempt in _baseline_attempts(row):
            plan_metadata = attempt.get("plan_metadata")
            has_decomposition_telemetry = False
            if "planned_decomposition" in attempt:
                planned_decomposition_attempts += int(bool(attempt.get("planned_decomposition")))
                has_decomposition_telemetry = True
            elif isinstance(plan_metadata, dict):
                planned_decomposition_attempts += int(int(plan_metadata.get("depth") or 0) > 0)
                has_decomposition_telemetry = True
            if "executed_decomposition" in attempt:
                executed_decomposition_attempts += int(bool(attempt.get("executed_decomposition")))
                has_decomposition_telemetry = True
            has_task_detection_telemetry = False
            if "task_detection_call_count" in attempt:
                task_detection_call_total += int(attempt.get("task_detection_call_count") or 0)
                has_task_detection_telemetry = True
            has_repl_query_telemetry = False
            if "repl_llm_query_call_count" in attempt:
                repl_llm_query_call_total += int(attempt.get("repl_llm_query_call_count") or 0)
                has_repl_query_telemetry = True
            if int(attempt.get("llm_call_count") or 0) > 0 and not (
                has_decomposition_telemetry
                and has_task_detection_telemetry
                and has_repl_query_telemetry
            ):
                unknown_attempts += 1
    return {
        "planned_decomposition_attempts": planned_decomposition_attempts,
        "executed_decomposition_attempts": executed_decomposition_attempts,
        "task_detection_call_total": task_detection_call_total,
        "repl_llm_query_call_total": repl_llm_query_call_total,
        "unknown_attempts": unknown_attempts,
    }


def _derive_rlm_outcome(attempt: dict[str, Any]) -> str:
    explicit = str(attempt.get("outcome_classification") or "").strip()
    if explicit:
        return explicit
    candidate_status = str(attempt.get("candidate_status") or "").strip()
    candidate_source = str(attempt.get("candidate_source") or "").strip()
    stop_reason = str(attempt.get("stop_reason") or "").strip()
    if candidate_status == "passed":
        if candidate_source == "final_answer_salvage":
            return "success_final_answer_salvage"
        return "success_workspace_write"
    if candidate_status == "invalid_workspace_write":
        return "failure_invalid_workspace_write"
    if candidate_status == "syntax_error":
        return "failure_syntax_error"
    if candidate_status == "workspace_unchanged":
        if stop_reason in _GUARD_STOP_REASONS:
            return "failure_search_exhausted"
        return "failure_workspace_unchanged"
    return ""


def _rlm_outcome_stats(rows: list[dict[str, Any]]) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    unknown_attempts = 0
    for row in rows:
        if str(row.get("condition") or "").strip() != "rlm":
            continue
        for attempt in _baseline_attempts(row):
            outcome = _derive_rlm_outcome(attempt)
            if outcome:
                counts[outcome] += 1
            elif (
                str(attempt.get("candidate_status") or "").strip()
                or str(attempt.get("stop_reason") or "").strip()
            ):
                unknown_attempts += 1
    return counts, unknown_attempts


def _baseline_guard_stop_reasons(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for attempt in _baseline_attempts(row):
            guard_name = str(attempt.get("guard_name") or "").strip()
            if guard_name:
                counts[guard_name] += 1
                continue
            stop_reason = str(attempt.get("stop_reason") or "").strip()
            if stop_reason in _GUARD_STOP_REASONS:
                counts[stop_reason] += 1
    return counts


def _lower_bound_harmed_cases(
    rows: list[dict[str, Any]],
    *,
    finish_reason_getter,
) -> int:
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = _condition(row)
        case_id = str(row.get("case_id") or "").strip()
        if not condition or not case_id:
            continue
        by_case[(condition, case_id)].append(row)

    harmed = 0
    for case_rows in by_case.values():
        if any(_is_success(row) for row in case_rows):
            continue
        finish_reasons = [finish_reason_getter(row) for row in case_rows]
        known = [value for value in finish_reasons if value]
        if known and len(known) == len(case_rows) and all(value == "length" for value in known):
            harmed += 1
    return harmed


def summarize_rows(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    attempt_finish_reasons = Counter()
    planner_finish_reasons = Counter()
    attempt_unknown = 0
    planner_unknown = 0
    for row in rows:
        finish_reason = _attempt_finish_reason(row)
        if finish_reason:
            attempt_finish_reasons[finish_reason] += 1
        elif _expects_attempt_finish_reason(row):
            attempt_unknown += 1

        planner_finish_reason = _planner_finish_reason(row)
        if planner_finish_reason:
            planner_finish_reasons[planner_finish_reason] += 1
        elif _expects_planner_finish_reason(row):
            planner_unknown += 1

    teacher_finish_reasons = _stage_finish_reasons(rows, "teacher_finish_reason")
    learner_finish_reasons = _stage_finish_reasons(rows, "learner_finish_reason")
    baseline_attempt_stop_reasons = _baseline_attempt_stop_reasons(rows)
    (
        baseline_call_finish_reasons,
        baseline_call_finish_unknown_attempts,
        baseline_call_length_finish_total,
        baseline_compat_recovery_total,
    ) = _baseline_call_finish_reason_stats(rows)
    baseline_guard_stop_reasons = _baseline_guard_stop_reasons(rows)
    lambda_stats = _lambda_attempt_stats(rows)
    rlm_outcomes, rlm_outcome_unknown_attempts = _rlm_outcome_stats(rows)
    prompt_budget_overflow_rows = sum(1 for row in rows if _prompt_budget_overflow_chars(row) > 0)
    compaction_trigger_rows = sum(1 for row in rows if _row_has_compaction_trigger(row))

    case_ids = {
        str(row.get("case_id") or "").strip()
        for row in rows
        if str(row.get("case_id") or "").strip()
    }
    condition_values = {
        str(row.get("condition") or "").strip()
        for row in rows
        if str(row.get("condition") or "").strip()
    }

    unknown_reasons: list[str] = []
    if attempt_unknown:
        unknown_reasons.append(
            f"{attempt_unknown} attempt rows do not expose a top-level finish_reason"
        )
    if planner_unknown:
        unknown_reasons.append(
            f"{planner_unknown} attempt rows do not expose planner finish_reason"
        )
    teacher_unknown = _count_stage_unknown(rows, "teacher_finish_reason")
    learner_unknown = _count_stage_unknown(rows, "learner_finish_reason")
    if teacher_unknown:
        unknown_reasons.append(
            f"{teacher_unknown} INTERVENOR teacher calls are missing finish_reason telemetry"
        )
    if learner_unknown:
        unknown_reasons.append(
            f"{learner_unknown} INTERVENOR learner calls are missing finish_reason telemetry"
        )
    if baseline_call_finish_unknown_attempts:
        unknown_reasons.append(
            f"{baseline_call_finish_unknown_attempts} baseline attempts are missing per-call "
            "finish_reason telemetry"
        )
    if lambda_stats["unknown_attempts"]:
        unknown_reasons.append(
            f"{lambda_stats['unknown_attempts']} lambda-RLM attempts are insufficient to assess "
            "decomposition execution from artifact telemetry"
        )
    if rlm_outcome_unknown_attempts:
        unknown_reasons.append(
            f"{rlm_outcome_unknown_attempts} RLM attempts are missing terminal outcome "
            "classification"
        )

    return {
        "source": source,
        "row_count": len(rows),
        "case_count": len(case_ids),
        "conditions": sorted(condition_values),
        "attempt_finish_reason_distribution": dict(sorted(attempt_finish_reasons.items())),
        "attempt_finish_reason_unknown_rows": attempt_unknown,
        "planner_finish_reason_distribution": dict(sorted(planner_finish_reasons.items())),
        "planner_finish_reason_unknown_rows": planner_unknown,
        "teacher_finish_reason_distribution": dict(sorted(teacher_finish_reasons.items())),
        "teacher_finish_reason_unknown_calls": teacher_unknown,
        "learner_finish_reason_distribution": dict(sorted(learner_finish_reasons.items())),
        "learner_finish_reason_unknown_calls": learner_unknown,
        "baseline_attempt_stop_reason_distribution": dict(
            sorted(baseline_attempt_stop_reasons.items())
        ),
        "baseline_call_finish_reason_distribution": dict(
            sorted(baseline_call_finish_reasons.items())
        ),
        "baseline_call_finish_reason_unknown_attempts": baseline_call_finish_unknown_attempts,
        "baseline_call_length_finish_total": baseline_call_length_finish_total,
        "baseline_compat_recovery_total": baseline_compat_recovery_total,
        "baseline_guard_stop_distribution": dict(sorted(baseline_guard_stop_reasons.items())),
        "lambda_planned_decomposition_attempts": lambda_stats["planned_decomposition_attempts"],
        "lambda_executed_decomposition_attempts": lambda_stats["executed_decomposition_attempts"],
        "lambda_task_detection_call_total": lambda_stats["task_detection_call_total"],
        "lambda_repl_llm_query_call_total": lambda_stats["repl_llm_query_call_total"],
        "lambda_decomposition_unknown_attempts": lambda_stats["unknown_attempts"],
        "rlm_outcome_distribution": dict(sorted(rlm_outcomes.items())),
        "rlm_outcome_unknown_attempts": rlm_outcome_unknown_attempts,
        "prompt_budget_overflow_rows": prompt_budget_overflow_rows,
        "context_compaction_auto_overflow_triggered_rows": compaction_trigger_rows,
        "lower_bound_cases_harmed_by_final_caps": _lower_bound_harmed_cases(
            rows,
            finish_reason_getter=_attempt_finish_reason,
        ),
        "lower_bound_cases_harmed_by_planner_caps": _lower_bound_harmed_cases(
            rows,
            finish_reason_getter=_planner_finish_reason,
        ),
        "unknown_due_to_missing_telemetry": unknown_reasons,
    }


def summarize_paths(paths: list[Path]) -> list[dict[str, Any]]:
    return [summarize_rows(_load_rows(path), source=str(path)) for path in paths]


def _render_text_summary(summary: dict[str, Any]) -> str:
    lines = [f"Source: {summary['source']}"]
    lines.append(
        f"Rows: {summary['row_count']} | Cases: {summary['case_count']} | Conditions: {', '.join(summary['conditions']) or 'unknown'}"
    )
    lines.append(
        "Attempt finish_reason: "
        + _format_distribution(
            summary["attempt_finish_reason_distribution"],
            unknown=int(summary["attempt_finish_reason_unknown_rows"]),
        )
    )
    lines.append(
        "Planner finish_reason: "
        + _format_distribution(
            summary["planner_finish_reason_distribution"],
            unknown=int(summary["planner_finish_reason_unknown_rows"]),
        )
    )
    lines.append(
        "Teacher finish_reason: "
        + _format_distribution(
            summary["teacher_finish_reason_distribution"],
            unknown=int(summary["teacher_finish_reason_unknown_calls"]),
        )
    )
    lines.append(
        "Learner finish_reason: "
        + _format_distribution(
            summary["learner_finish_reason_distribution"],
            unknown=int(summary["learner_finish_reason_unknown_calls"]),
        )
    )
    lines.append(
        "Baseline attempt stop_reason: "
        + _format_distribution(summary["baseline_attempt_stop_reason_distribution"], unknown=0)
    )
    lines.append(
        "Baseline call finish_reason: "
        + _format_distribution(
            summary["baseline_call_finish_reason_distribution"],
            unknown=int(summary["baseline_call_finish_reason_unknown_attempts"]),
        )
    )
    lines.append(
        "Baseline call saturation: "
        f"length_calls={summary['baseline_call_length_finish_total']} | "
        f"compat_recoveries={summary['baseline_compat_recovery_total']}"
    )
    if isinstance(summary.get("baseline_guard_stop_distribution"), dict):
        lines.append(
            "Baseline guard stops: "
            + _format_distribution(summary["baseline_guard_stop_distribution"], unknown=0)
        )
    lines.append(
        "lambda-RLM decomposition: "
        f"planned={summary['lambda_planned_decomposition_attempts']} | "
        f"executed={summary['lambda_executed_decomposition_attempts']} | "
        f"task_detection_calls={summary['lambda_task_detection_call_total']} | "
        f"repl_llm_query_calls={summary['lambda_repl_llm_query_call_total']} | "
        f"unknown={summary['lambda_decomposition_unknown_attempts']}"
    )
    lines.append(
        "RLM outcomes: "
        + _format_distribution(
            summary["rlm_outcome_distribution"],
            unknown=int(summary["rlm_outcome_unknown_attempts"]),
        )
    )
    lines.append(
        "Prompt overflow rows: "
        f"{summary['prompt_budget_overflow_rows']} | "
        "Compaction-trigger rows: "
        f"{summary['context_compaction_auto_overflow_triggered_rows']}"
    )
    lines.append(
        "Lower-bound harmed cases: "
        f"final_caps={summary['lower_bound_cases_harmed_by_final_caps']} "
        f"planner_caps={summary['lower_bound_cases_harmed_by_planner_caps']}"
    )
    unknown = summary["unknown_due_to_missing_telemetry"]
    if isinstance(unknown, list) and unknown:
        lines.append("Unknown bucket:")
        lines.extend(f"- {item}" for item in unknown)
    return "\n".join(lines)


def _format_distribution(distribution: Any, *, unknown: int) -> str:
    if not isinstance(distribution, dict) or not distribution:
        return f"none reported (unknown={unknown})"
    parts = [f"{key}={value}" for key, value in sorted(distribution.items())]
    parts.append(f"unknown={unknown}")
    return ", ".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Eval JSONL result files to summarize.")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths = [Path(raw).expanduser() for raw in args.paths]
    summaries = summarize_paths(paths)
    if args.format == "json":
        print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("\n\n".join(_render_text_summary(summary) for summary in summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
