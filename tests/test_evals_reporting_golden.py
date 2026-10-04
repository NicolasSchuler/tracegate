"""Golden-file tests for eval_summary and report_run_metrics (2026-10-04 review, Phase 2).

The fixture has the shapes the review found mishandled: unequal case sets (STA-03),
missing token usage (STA-01, STA-09) and several runs of the same case (STA-02).
Regenerate the golden files with ``UPDATE_GOLDEN=1 pytest tests/test_evals_reporting_golden.py``
after an intended output change, and review the diff.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from evals import _json as json
from evals import eval_summary, report_run_metrics
from tests.test_evals_analyze_grouping import _record

GOLDEN_DIR = Path(__file__).parent / "golden"
RUN_ID = "golden-run"


def _records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_no in (1, 2):
        for idx in range(6):
            conditions = ["base_repair", "layered_adaptive_llm"]
            if idx >= 4:
                conditions = ["base_repair"]  # unequal case sets
            for condition in conditions:
                success = (idx + run_no) % 3 == 0 if condition == "base_repair" else idx % 2 == 0
                record = _record(
                    run_id=RUN_ID,
                    case_id=f"case{idx}",
                    condition=condition,
                    category="logic_error" if idx % 2 else "hidden_state",
                    difficulty="easy",
                    snapshot_dependency="required" if idx < 3 else "helpful",
                    attempt=run_no,  # two attempts per case stand in for repeated work
                    success=success,
                )
                tokens = 100 * (idx + 1) + (50 if condition != "base_repair" else 0)
                record["attempt_metadata"].update(
                    {"tokens_prompt": tokens - 10, "tokens_completion": 10, "tokens_total": tokens}
                )
                if idx == 1 and condition == "layered_adaptive_llm":
                    record["attempt_metadata"]["token_usage_available"] = False
                records.append(record)
    # A second run of the same cases (STA-02).
    second_run = []
    for record in records:
        if record["attempt"] == 1:
            copy = json.loads(json.dumps(record))
            copy["run_id"] = f"{RUN_ID}-b"
            copy["run_metadata"]["run_id"] = f"{RUN_ID}-b"
            second_run.append(copy)
    return records + second_run


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / f"{RUN_ID}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _records()), encoding="utf-8")
    return path


def _check_golden(name: str, output: str, tmp_path: Path) -> None:
    normalized = output.replace(str(tmp_path), "<tmp>")
    golden = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(normalized, encoding="utf-8")
    assert golden.is_file(), f"missing golden file {golden}; run with UPDATE_GOLDEN=1"
    assert normalized == golden.read_text(encoding="utf-8")


@pytest.mark.filterwarnings("ignore")
def test_eval_summary_golden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_fixture(tmp_path)
    assert eval_summary.main([RUN_ID, "--results-dir", str(tmp_path)]) == 0
    _check_golden("eval_summary_phase2.txt", capsys.readouterr().out, tmp_path)


@pytest.mark.filterwarnings("ignore")
def test_report_run_metrics_golden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_fixture(tmp_path)
    assert report_run_metrics.main(["--jsonl", str(path), "--bootstrap", "500", "--seed", "3"]) == 0
    _check_golden("report_run_metrics_phase2.txt", capsys.readouterr().out, tmp_path)
