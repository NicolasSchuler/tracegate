"""Re-analysis audit for existing eval results (2026-10-04 review, Phase 2.4).

Usage::

    python -m evals.reanalysis_audit evals/results/*.jsonl \\
        [--a base_repair] [--b layered_adaptive_llm] \\
        [--swebench-run-dir evals/swebench_runs/<id> ...] [--json audit.json]

Part 1 recomputes headline metrics with the corrected analysis and, from the same
records, with the semantics the analysis had before the review, and prints the change.
Part 2 runs the data audits listed under "Impact on existing results" in
``docs/2026-10-04-code-review-findings.md``. The tool only reads results; it never edits
them or any published number.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import _json as json
from .analyze_results import (
    ADAPTIVE_BUDGET_ACTIONS,
    ADAPTIVE_ONLINE_ACTIONS,
    _adaptive_ope_samples,
    _case_success_rates,
    _derive_case_condition_rows,
    _expand_paths,
    _load_and_group_records,
    _matched_uplift,
    _run_statistical_analysis,
)
from .eval_types import (
    EVAL_RECORD_SCHEMA_VERSION,
    HELPER_RESTRICTED_STAGE_A_ACTIONS,
    normalize_condition_name,
)
from .stats import analyze_paired_results, exact_binomial_p_value, mcnemar_test, power_report

_CONFIG_AND_TEST_NAMES = ("conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
_EXAMPLE_LIMIT = 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(paths: Iterable[str]) -> list[dict[str, Any]]:
    """Every JSON object line of *paths*, including telemetry rows."""
    records: list[dict[str, Any]] = []
    for src in _expand_paths(list(paths)):
        try:
            text = Path(src).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _patch_meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("patch_meta")
    return value if isinstance(value, dict) else {}


def _attempt_meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("attempt_metadata")
    return value if isinstance(value, dict) else {}


def _run_meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("run_metadata")
    return value if isinstance(value, dict) else {}


def _schema(record: dict[str, Any]) -> int:
    try:
        return int(record.get("schema_version") or 0)
    except (TypeError, ValueError):
        return 0


def _record_id(record: dict[str, Any]) -> str:
    return (
        f"{record.get('run_id')}/{record.get('case_id')}/"
        f"{normalize_condition_name(record.get('condition'))}#{record.get('attempt')}"
    )


def _audit(finding: str, count: int, examples: list[str], **detail: Any) -> dict[str, Any]:
    return {"finding": finding, "count": count, "examples": examples[:_EXAMPLE_LIMIT], **detail}


# ---------------------------------------------------------------------------
# Part 1: headline metrics, corrected vs pre-review semantics
# ---------------------------------------------------------------------------


def _legacy_mcnemar_p(pairs: list[tuple[bool, bool]]) -> float:
    """McNemar as computed before STA-23: chi-square with continuity above 24 pairs."""
    n_01 = sum(1 for a, b in pairs if not a and b)
    n_10 = sum(1 for a, b in pairs if a and not b)
    n = n_01 + n_10
    if n == 0:
        return 1.0
    if n < 25:
        return exact_binomial_p_value(n_01, n)
    statistic = (abs(n_01 - n_10) - 1) ** 2 / n
    return math.erfc(math.sqrt(statistic / 2.0))


def _legacy_power(n_discordant: int, uplift: float) -> float:
    """``power_report`` before STA-04."""
    if n_discordant == 0 or abs(uplift) < 1e-10:
        return 0.0
    denom = math.sqrt(1 - uplift**2) if abs(uplift) < 1.0 else 0.01
    z = math.sqrt(n_discordant) * abs(uplift) / denom - 1.96
    return 0.5 * math.erfc(-z / math.sqrt(2))


def headline_comparison(
    rows: list[dict[str, Any]], *, condition_a: str, condition_b: str
) -> dict[str, Any]:
    """Headline metrics with the corrected code and with the pre-review semantics."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    expected_cases = {str(row["case_id"]) for row in rows}

    conditions: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        tokens = [row.get("token_usage") for row in condition_rows]
        measured = [float(value) for value in tokens if value is not None]
        conditions[condition] = {
            "cases": len(condition_rows),
            "success_rate": sum(bool(row["success"]) for row in condition_rows)
            / len(condition_rows),
            "cases_missing": len(expected_cases - {str(r["case_id"]) for r in condition_rows}),
            "mean_tokens_corrected": (sum(measured) / len(measured)) if measured else None,
            "mean_tokens_legacy": sum(float(value or 0.0) for value in tokens)
            / len(condition_rows),
            "token_covered_cases": len(measured),
        }

    case_rates = {
        condition: _case_success_rates(condition_rows)
        for condition, condition_rows in by_condition.items()
    }
    a_stats = conditions.get(condition_a)
    b_stats = conditions.get(condition_b)
    uplift_legacy = (
        b_stats["success_rate"] - a_stats["success_rate"] if a_stats and b_stats else None
    )
    result: dict[str, Any] = {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "conditions": conditions,
        "uplift_legacy_unmatched": uplift_legacy,
        "uplift_corrected_matched": _matched_uplift(case_rates, condition_a, condition_b),
    }

    paired: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if str(row["condition"]) in (condition_a, condition_b):
            paired[(str(row["run_id"]), str(row["case_id"]))][str(row["condition"])] = bool(
                row["success"]
            )
    run_pairs = [
        (entry[condition_a], entry[condition_b])
        for entry in paired.values()
        if condition_a in entry and condition_b in entry
    ]
    corrected = analyze_paired_results(rows, condition_a=condition_a, condition_b=condition_b)
    if run_pairs and "error" not in corrected:
        legacy_mcnemar = mcnemar_test(run_pairs)
        legacy_uplift = (sum(b for _, b in run_pairs) - sum(a for a, _ in run_pairs)) / len(
            run_pairs
        )
        corrected_mcnemar = corrected["overall_mcnemar"]
        result.update(
            {
                "pairs_legacy_run_level": len(run_pairs),
                "pairs_corrected_case_level": corrected["n_pairs"],
                "mcnemar_p_legacy": _legacy_mcnemar_p(run_pairs),
                "mcnemar_p_corrected": corrected_mcnemar["p_value"],
                "power_legacy": _legacy_power(legacy_mcnemar["n_discordant"], legacy_uplift),
                "power_corrected": power_report(
                    corrected["n_decided_pairs"],
                    corrected_mcnemar["n_01"],
                    corrected_mcnemar["n_10"],
                )["approximate_power"],
            }
        )
    stats = _run_statistical_analysis(
        rows, aggregation="majority_vote", pair_a=condition_a, pair_b=condition_b
    )
    pairwise = [entry for entry in stats.get("pairwise", []) if "error" not in entry]
    result["pairwise_significant_uncorrected"] = sum(
        1 for entry in pairwise if entry.get("mcnemar_significant")
    )
    result["pairwise_significant_holm"] = sum(1 for entry in pairwise if entry.get("holm_rejected"))
    result["pairwise_tests"] = len(pairwise)
    return result


# ---------------------------------------------------------------------------
# Part 2: data audits
# ---------------------------------------------------------------------------


def _touches_tests_or_config(patch_meta: dict[str, Any]) -> bool:
    if patch_meta.get("patch_touches_tests") or patch_meta.get("test_surface_modified"):
        return True
    paths = patch_meta.get("patch_file_paths")
    if not isinstance(paths, list):
        return False
    for raw in paths:
        name = Path(str(raw)).name
        if name in _CONFIG_AND_TEST_NAMES or name.startswith("test_") or name.endswith("_test.py"):
            return True
    return False


def data_audits(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The checks listed under "Impact on existing results"."""
    attempts = [r for r in records if str(r.get("record_type") or "attempt") != "infra_event"]
    audits: dict[str, Any] = {}

    touched = [
        _record_id(r)
        for r in attempts
        if bool(r.get("success")) and _touches_tests_or_config(_patch_meta(r))
    ]
    audits["successes_touching_tests_or_config"] = _audit(
        "ATT-01: successful attempts whose patch touched tests or pytest config",
        len(touched),
        touched,
    )

    fallback = [
        _record_id(r)
        for r in attempts
        if _patch_meta(r).get("stage_a_used_fallback")
        and _patch_meta(r).get("stage_a_reason") == "no_valid_stage_a_requests"
    ]
    audits["stage_a_whole_file_fallback"] = _audit(
        "ATT-03: attempts that got the whole-file fallback (no valid Stage-A requests)",
        len(fallback),
        fallback,
    )
    seeded_restricted = [
        _record_id(r)
        for r in attempts
        if _schema(r) < EVAL_RECORD_SCHEMA_VERSION
        and _patch_meta(r).get("adaptive_helper_materialized_context")
        and str(_patch_meta(r).get("adaptive_helper_action_executed") or "")
        in HELPER_RESTRICTED_STAGE_A_ACTIONS
    ]
    audits["seed_bundle_on_restricted_actions"] = _audit(
        "ATT-04: pre-v7 adaptive attempts with a restricted helper action, which still got "
        "the whole-file seed bundle",
        len(seeded_restricted),
        seeded_restricted,
    )

    configs: dict[str, dict[str, Any]] = {}
    for record in attempts:
        config = _run_meta(record).get("patcher_config")
        if isinstance(config, dict):
            configs.setdefault(str(record.get("run_id") or ""), config)
    full_reveal = sorted(
        run_id
        for run_id, config in configs.items()
        if str(config.get("adaptive_reveal_mode") or "full") == "full"
        and any(
            _schema(r) < EVAL_RECORD_SCHEMA_VERSION
            for r in attempts
            if str(r.get("run_id") or "") == run_id
        )
    )
    audits["full_reveal_runs"] = _audit(
        "ATT-06: pre-v7 runs in full reveal mode, where the snapshot was always in the prompt",
        len(full_reveal),
        full_reveal,
    )
    decays = {
        run_id: (config.get("adaptive_online_decay"), config.get("adaptive_budget_online_decay"))
        for run_id, config in configs.items()
        if "adaptive_online_decay" in config or "adaptive_budget_online_decay" in config
    }
    decayed = sorted(
        run_id
        for run_id, values in decays.items()
        if any(isinstance(v, (int, float)) and v < 1.0 for v in values)
    )
    audits["policy_decay_below_one"] = _audit(
        "POL-01: runs whose online or budget policy decay is below 1.0 (0.95 barely learns)",
        len(decayed),
        decayed,
        decay_by_run={run_id: list(values) for run_id, values in sorted(decays.items())},
    )

    shadow = missing = 0
    for prefix, actions in (
        ("adaptive_online", ADAPTIVE_ONLINE_ACTIONS),
        ("adaptive_budget", ADAPTIVE_BUDGET_ACTIONS),
    ):
        valid_actions = {str(action) for action in actions}
        for sample in _adaptive_ope_samples(
            records=attempts, prefix=prefix, valid_actions=valid_actions
        ):
            shadow += bool(sample.get("propensity_shadow_corrected"))
            missing += bool(sample.get("propensity_missing"))
    audits["ope_propensities"] = _audit(
        "POL-03, STA-12: OPE samples with a wrong shadow propensity or none at all",
        shadow + missing,
        [],
        shadow_corrected=shadow,
        missing_propensity=missing,
    )

    deterministic = sorted(
        {
            str(r.get("run_id") or "")
            for r in attempts
            if _run_meta(r).get("deterministic") is True and _schema(r) < EVAL_RECORD_SCHEMA_VERSION
        }
    )
    audits["deterministic_runs_pre_fix"] = _audit(
        "ORC-01, ORC-02: pre-v7 runs marked deterministic (pytest wasn't hash-seeded)",
        len(deterministic),
        deterministic,
    )

    runs_per_case: Counter[tuple[str, str]] = Counter(
        (str(row["case_id"]), str(row["condition"])) for row in rows
    )
    pooled = sorted(f"{case}/{cond}" for (case, cond), n in runs_per_case.items() if n > 1)
    audits["cases_with_repeated_runs"] = _audit(
        "STA-02: case-condition pairs with several runs (pooled as independent pairs before)",
        len(pooled),
        pooled,
    )
    expected = {str(row["case_id"]) for row in rows}
    by_condition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_condition[str(row["condition"])].add(str(row["case_id"]))
    missing_cases = {
        condition: len(expected - cases)
        for condition, cases in sorted(by_condition.items())
        if expected - cases
    }
    audits["unmatched_case_sets"] = _audit(
        "STA-03: conditions that didn't run every case in the input",
        len(missing_cases),
        sorted(missing_cases),
        cases_missing=missing_cases,
    )

    unmeasured = Counter(
        normalize_condition_name(r.get("condition"))
        for r in attempts
        if _attempt_meta(r).get("token_usage_available") is False
    )
    audits["attempts_without_token_usage"] = _audit(
        "STA-01, STA-09: attempts without token usage (counted as 0 tokens before)",
        sum(unmeasured.values()),
        [],
        by_condition=dict(sorted(unmeasured.items())),
    )

    ldb_rows = [row for row in rows if row["condition"] == "ldb"]
    audits["ldb_cases"] = _audit(
        "BAS-01: LDB case counts (only cases it can convert are comparable)",
        len(ldb_rows),
        [],
        unsupported=sum(1 for row in ldb_rows if row.get("baseline_supported") is False),
        records_without_support_flag=sum(
            1
            for r in attempts
            if normalize_condition_name(r.get("condition")) == "ldb"
            and "baseline_supported" not in _patch_meta(r)
        ),
    )

    retry_gate = [
        _record_id(r) for r in attempts if str(r.get("note") or "") == "retry_gate_no_new_evidence"
    ]
    audits["retry_gate_scored_attempts"] = _audit(
        "ATT-05: retry-gate blocks recorded as scored attempts",
        len(retry_gate),
        retry_gate,
    )
    forced = [
        _record_id(r)
        for r in attempts
        if _patch_meta(r).get("runner_forced_patch_attempt")
        and str(r.get("note") or "") == "tests_still_failing"
    ]
    audits["forced_patch_as_test_failure"] = _audit(
        "ATT-07: forced marker patches recorded as tests_still_failing",
        len(forced),
        forced,
    )
    audits["schema_versions"] = _audit(
        f"Records older than schema {EVAL_RECORD_SCHEMA_VERSION} predate the review fixes",
        sum(1 for r in attempts if _schema(r) < EVAL_RECORD_SCHEMA_VERSION),
        [],
        by_version=dict(sorted(Counter(_schema(r) for r in attempts).items())),
    )
    return audits


def swebench_audit(run_dir: Path) -> dict[str, Any]:
    """DAT-01, DAT-02, DAT-03, DAT-08, DAT-18 checks for one SWE-Bench run directory."""
    metadata_path = run_dir / "run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"run_dir": str(run_dir), "error": "unreadable run_metadata.json"}
    conditions = [str(c) for c in metadata.get("conditions") or []]
    policy = metadata.get("prompt_policy")
    policy = policy if isinstance(policy, dict) else None
    empty_patches: dict[str, int] = {}
    for condition in conditions:
        path = run_dir / condition / "predictions.jsonl"
        if not path.is_file():
            continue
        empty_patches[condition] = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not str(json.loads(line).get("model_patch") or "").strip()
        )
    return {
        "run_dir": str(run_dir),
        "conditions": conditions,
        "llmdebug_without_snapshot_dir": "llmdebug" in conditions
        and not (policy or {}).get("snapshot_dir"),
        "prompt_policy_recorded": policy is not None,
        "hints_or_test_ids_in_prompt": policy is None
        or bool(policy.get("include_hints") or policy.get("include_test_ids")),
        "empty_patches": empty_patches,
        "generation_meta_recorded": any(
            (run_dir / c / "generation_meta.jsonl").is_file() for c in conditions
        ),
    }


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_text(report: dict[str, Any]) -> str:
    """Human-readable audit report."""
    headline = report["headline"]
    lines = [
        "Re-analysis audit (2026-10-04 review)",
        f"- input records: {report['records']}, case-condition rows: {report['rows']}",
        "",
        f"Headline metrics: corrected vs pre-review semantics "
        f"({headline['condition_a']} -> {headline['condition_b']})",
    ]
    for condition, stats in headline["conditions"].items():
        lines.append(
            f"- {condition}: success={_fmt(stats['success_rate'])} over {stats['cases']} cases "
            f"(missing {stats['cases_missing']}); mean tokens corrected="
            f"{_fmt(stats['mean_tokens_corrected'])} vs legacy={_fmt(stats['mean_tokens_legacy'])}"
            f" (measured {stats['token_covered_cases']}/{stats['cases']})"
        )
    lines.append(
        f"- uplift: legacy unmatched={_fmt(headline['uplift_legacy_unmatched'])}, "
        f"corrected matched={_fmt(headline['uplift_corrected_matched'])}  [STA-03]"
    )
    if "mcnemar_p_legacy" in headline:
        lines.append(
            f"- paired test: legacy {headline['pairs_legacy_run_level']} run-level pairs "
            f"p={_fmt(headline['mcnemar_p_legacy'])}, corrected "
            f"{headline['pairs_corrected_case_level']} cases "
            f"p={_fmt(headline['mcnemar_p_corrected'])}  [STA-02, STA-23]"
        )
        lines.append(
            f"- power: legacy={_fmt(headline['power_legacy'])}, "
            f"corrected={_fmt(headline['power_corrected'])}  [STA-04]"
        )
    lines.append(
        f"- pairwise tests significant: {headline['pairwise_significant_uncorrected']} "
        f"uncorrected vs {headline['pairwise_significant_holm']} after Holm, of "
        f"{headline['pairwise_tests']}  [STA-06]"
    )
    lines.extend(["", "Data audits"])
    for name, audit in report["audits"].items():
        extra = {
            key: value
            for key, value in audit.items()
            if key not in {"finding", "count", "examples"}
        }
        lines.append(f"- {name}: {audit['count']} — {audit['finding']}")
        if audit["examples"]:
            lines.append(f"    examples: {', '.join(audit['examples'])}")
        if extra:
            lines.append(f"    {json.dumps(extra, sort_keys=True)}")
    for swebench in report.get("swebench", []):
        lines.extend(["", f"SWE-Bench run {swebench['run_dir']}"])
        lines.extend(f"- {key}: {value}" for key, value in swebench.items() if key != "run_dir")
    return "\n".join(lines)


def build_report(
    paths: list[str], *, condition_a: str, condition_b: str, swebench_dirs: list[Path]
) -> dict[str, Any]:
    records = load_records(paths)
    grouped, _input_errors = _load_and_group_records(paths)
    rows = _derive_case_condition_rows(grouped)
    return {
        "records": len(records),
        "rows": len(rows),
        "headline": headline_comparison(rows, condition_a=condition_a, condition_b=condition_b),
        "audits": data_audits(records, rows),
        "swebench": [swebench_audit(path) for path in swebench_dirs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-analysis audit for existing eval results (2026-10-04 review)."
    )
    parser.add_argument("paths", nargs="+", help="Eval result JSONL files (globs ok).")
    parser.add_argument("--a", default="base_repair", help="Condition A of the headline pair.")
    parser.add_argument("--b", default="layered_adaptive_llm", help="Condition B.")
    parser.add_argument(
        "--swebench-run-dir",
        action="append",
        default=[],
        type=Path,
        help="SWE-Bench run directory to check (repeatable).",
    )
    parser.add_argument("--json", default="", help="Also write the report as JSON here.")
    args = parser.parse_args(argv)
    report = build_report(
        list(args.paths),
        condition_a=str(args.a),
        condition_b=str(args.b),
        swebench_dirs=list(args.swebench_run_dir),
    )
    if not report["rows"]:
        print("No case-condition rows found.")
        return 1
    print(render_text(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
