"""Phase 2.4: the re-analysis audit tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import _json as json
from evals import reanalysis_audit as audit
from tests.test_evals_analyze_grouping import _record


def _rec(case_id: str, condition: str, success: bool, **overrides: Any) -> dict[str, Any]:
    return _record(
        run_id=overrides.pop("run_id", "r1"),
        case_id=case_id,
        condition=condition,
        category="logic_error",
        difficulty="easy",
        snapshot_dependency="helpful",
        attempt=overrides.pop("attempt", 1),
        success=success,
        **overrides,
    )


def _write(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> Path:
    records = [
        # c1: B solves by editing tests; both runs present
        _rec("c1", "base_repair", False),
        _rec("c1", "layered_adaptive_llm", True, patch_meta={"patch_touches_tests": True}),
        _rec("c1", "base_repair", False, run_id="r2"),
        _rec("c1", "layered_adaptive_llm", True, run_id="r2"),
        # c2: only base ran (unmatched case set); no token usage; deterministic run
        _rec("c2", "base_repair", True),
        # c3: adaptive attempt with the whole-file fallback, a restricted action and a
        # retry-gate block that consumed the attempt
        _rec(
            "c3",
            "layered_adaptive_llm",
            False,
            note="retry_gate_no_new_evidence",
            patch_meta={
                "stage_a_used_fallback": True,
                "stage_a_reason": "no_valid_stage_a_requests",
                "adaptive_helper_materialized_context": True,
                "adaptive_helper_action_executed": "file_only",
            },
        ),
        _rec("c3", "base_repair", False),
    ]
    for record in records:
        record["run_metadata"]["deterministic"] = True
        record["run_metadata"]["patcher_config"] = {
            "adaptive_online_decay": 0.95,
            "adaptive_budget_online_decay": 1.0,
            "adaptive_reveal_mode": "full",
        }
    records[4]["attempt_metadata"]["token_usage_available"] = False
    records.append({"record_type": "infra_event", "run_id": "r1", "case_id": "c9"})
    return _write(tmp_path, records)


def test_audit_flags_each_issue_in_old_records(tmp_path: Path) -> None:
    report = audit.build_report(
        [str(_fixture(tmp_path))],
        condition_a="base_repair",
        condition_b="layered_adaptive_llm",
        swebench_dirs=[],
    )
    audits = report["audits"]
    assert audits["successes_touching_tests_or_config"]["count"] == 1
    assert audits["stage_a_whole_file_fallback"]["count"] == 1
    assert audits["seed_bundle_on_restricted_actions"]["count"] == 1
    assert audits["full_reveal_runs"]["examples"] == ["r1", "r2"]
    assert audits["policy_decay_below_one"]["examples"] == ["r1", "r2"]
    assert audits["deterministic_runs_pre_fix"]["count"] == 2
    assert audits["cases_with_repeated_runs"]["examples"] == [
        "c1/base_repair",
        "c1/layered_adaptive_llm",
    ]
    assert audits["unmatched_case_sets"]["cases_missing"] == {"layered_adaptive_llm": 1}
    assert audits["attempts_without_token_usage"]["by_condition"] == {"base_repair": 1}
    assert audits["retry_gate_scored_attempts"]["count"] == 1
    assert audits["schema_versions"]["by_version"] == {5: 7}


def test_headline_reports_corrected_and_legacy_values(tmp_path: Path) -> None:
    report = audit.build_report(
        [str(_fixture(tmp_path))],
        condition_a="base_repair",
        condition_b="layered_adaptive_llm",
        swebench_dirs=[],
    )
    headline = report["headline"]
    # Legacy: rates over unmatched sets (B 2/3 rows, A 1/4 rows); corrected: c1 and c3 only.
    assert headline["uplift_legacy_unmatched"] == pytest.approx(2 / 3 - 1 / 4)
    assert headline["uplift_corrected_matched"] == pytest.approx(0.5)
    assert headline["pairs_legacy_run_level"] == 3
    assert headline["pairs_corrected_case_level"] == 2
    base = headline["conditions"]["base_repair"]
    assert base["token_covered_cases"] == 3
    assert base["mean_tokens_legacy"] <= base["mean_tokens_corrected"]


def test_cli_prints_report_and_writes_json(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "swebench_run"
    (run_dir / "llmdebug").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"conditions": ["baseline", "llmdebug"]}), encoding="utf-8"
    )
    (run_dir / "llmdebug" / "predictions.jsonl").write_text(
        json.dumps({"instance_id": "i1", "model_patch": "", "model_name_or_path": "m"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "audit.json"

    code = audit.main(
        [str(_fixture(tmp_path)), "--swebench-run-dir", str(run_dir), "--json", str(out)]
    )

    assert code == 0
    text = capsys.readouterr().out
    assert "Headline metrics: corrected vs pre-review semantics" in text
    assert "successes_touching_tests_or_config: 1" in text
    swebench = json.loads(out.read_text(encoding="utf-8"))["swebench"][0]
    assert swebench["llmdebug_without_snapshot_dir"] is True
    assert swebench["hints_or_test_ids_in_prompt"] is True
    assert swebench["empty_patches"] == {"llmdebug": 1}


def test_cli_without_rows_returns_one(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert audit.main([str(empty)]) == 1
    assert "No case-condition rows" in capsys.readouterr().out
