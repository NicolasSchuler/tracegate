"""Tests for two-pass eval summary generation and aggregation."""

from __future__ import annotations

from pathlib import Path

from evals.eval_two_pass_summary import main
from llmdebug import _json as json


def _record(
    *,
    run_id: str,
    case_id: str,
    condition: str,
    success: bool,
    schema_version: int = 5,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": 1,
        "success": success,
        "category": "logic_error",
        "difficulty": "easy",
        "snapshot_dependency": "required",
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": "2026-03-01T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.11.0",
            "platform": "test",
            "llmdebug_version": "2.11.4",
            "patcher_name": "openai",
            "patcher_config_fingerprint": "fp",
        },
        "attempt_metadata": {
            "attempt": 1,
            "max_attempts": 3,
            "attempt_ordinal_label": "1/3",
            "llm_call_count": 1,
            "tokens_prompt": 10,
            "tokens_completion": 2,
            "tokens_total": 12,
            "retry_history_budget_chars": 12000,
            "retry_history_budget_applied": False,
            "prompt_sha256": "prompt-hash",
            "prompt_path": None,
            "evidence_sha256": "evidence-hash",
            "evidence_path": None,
        },
        "model_metadata": {
            "backend": "openai",
            "name": "openai",
            "base_url": "http://127.0.0.1:12000/v1",
            "model": "dummy",
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_mode": "edits",
            "api_compat_seed_requested": None,
            "api_compat_seed_acknowledged": False,
        },
        "cumulative_runtime_sec": 1.0,
        "note": "success" if success else "tests_still_failing",
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_two_pass_summary_missing_file(tmp_path: Path, capsys) -> None:
    rc = main(["base", "rescue", "--results-dir", str(tmp_path)])
    assert rc == 1
    assert "results file not found" in capsys.readouterr().out


def test_two_pass_summary_strict_schema_rejection(tmp_path: Path, capsys) -> None:
    _write_jsonl(
        tmp_path / "base.jsonl",
        [_record(run_id="base", case_id="case-a", condition="base_repair", success=False)],
    )
    _write_jsonl(
        tmp_path / "rescue.jsonl",
        [
            _record(
                run_id="rescue",
                case_id="case-a",
                condition="layered_adaptive_llm",
                success=True,
                schema_version=2,
            )
        ],
    )
    rc = main(["base", "rescue", "--results-dir", str(tmp_path)])
    assert rc == 2
    assert "unsupported schema_version=2" in capsys.readouterr().out


def test_two_pass_summary_computes_rescue_uplift(tmp_path: Path, capsys) -> None:
    _write_jsonl(
        tmp_path / "base.jsonl",
        [
            _record(run_id="base", case_id="case-a", condition="base_repair", success=False),
            _record(run_id="base", case_id="case-b", condition="base_repair", success=False),
            _record(run_id="base", case_id="case-c", condition="base_repair", success=True),
        ],
    )
    _write_jsonl(
        tmp_path / "rescue.jsonl",
        [
            _record(
                run_id="rescue",
                case_id="case-a",
                condition="layered_adaptive_llm",
                success=True,
            ),
            _record(
                run_id="rescue",
                case_id="case-b",
                condition="layered_adaptive_llm",
                success=False,
            ),
        ],
    )
    rc = main(["base", "rescue", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "- success: 1/3 (33.3%)" in out
    assert "- attempted failed cases: 2/2 (100.0%)" in out
    assert "1/2 on attempted (50.0%), 1/2 on failed pool (50.0%)" in out
    assert "- final success: 2/3 (66.7%)" in out
    assert "- absolute uplift vs baseline: +33.3%" in out


def test_two_pass_summary_handles_no_failed_baseline(tmp_path: Path, capsys) -> None:
    _write_jsonl(
        tmp_path / "base.jsonl",
        [_record(run_id="base", case_id="case-a", condition="base_repair", success=True)],
    )
    _write_jsonl(
        tmp_path / "rescue.jsonl",
        [
            _record(
                run_id="rescue",
                case_id="case-a",
                condition="layered_adaptive_llm",
                success=True,
            )
        ],
    )
    rc = main(["base", "rescue", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "- failed cases: 0" in out
    assert "- attempted failed cases: 0/0 (n/a)" in out
    assert "- final success: 1/1 (100.0%)" in out
    assert "- absolute uplift vs baseline: +0.0%" in out


def test_two_pass_summary_rejects_mixed_run_ids_in_input_file(tmp_path: Path, capsys) -> None:
    _write_jsonl(
        tmp_path / "base.jsonl",
        [
            _record(run_id="base", case_id="case-a", condition="base_repair", success=False),
            _record(run_id="other", case_id="case-b", condition="base_repair", success=True),
        ],
    )
    _write_jsonl(
        tmp_path / "rescue.jsonl",
        [
            _record(
                run_id="rescue",
                case_id="case-a",
                condition="layered_adaptive_llm",
                success=True,
            )
        ],
    )
    rc = main(["base", "rescue", "--results-dir", str(tmp_path)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "must contain only run_id='base'" in out
