"""Tests for the dedicated failed-case extraction CLI."""

from __future__ import annotations

from pathlib import Path

from evals import extract_failed_cases
from llmdebug import _json as json


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extract_failed_cases_infers_single_condition_and_uses_default_output(
    tmp_path, capsys
) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "run.failed.txt"
    _write_jsonl(
        input_path,
        [
            {"case_id": "case_a", "condition": "base_repair", "attempt": 1, "success": False},
            {"case_id": "case_a", "condition": "base_repair", "attempt": 2, "success": True},
            {"case_id": "case_b", "condition": "base_repair", "attempt": 1, "success": False},
            {"case_id": "case_c", "condition": "base_repair", "attempt": 2, "success": False},
        ],
    )

    exit_code = extract_failed_cases.main([str(input_path)])

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_b\ncase_c\n"
    captured = capsys.readouterr().out
    assert "Extracted 2 failed case(s)" in captured
    assert f"Wrote case list: {output_path}" in captured


def test_extract_failed_cases_requires_explicit_condition_when_multiple_present(
    tmp_path, capsys
) -> None:
    input_path = tmp_path / "run.jsonl"
    _write_jsonl(
        input_path,
        [
            {"case_id": "case_a", "condition": "base_repair", "attempt": 1, "success": False},
            {"case_id": "case_b", "condition": "cot_repair", "attempt": 1, "success": False},
        ],
    )

    exit_code = extract_failed_cases.main([str(input_path)])

    assert exit_code == 2
    assert "Multiple conditions found in results JSONL" in capsys.readouterr().out


def test_extract_failed_cases_honors_explicit_condition_and_max_attempt(tmp_path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "failed.txt"
    _write_jsonl(
        input_path,
        [
            {"case_id": "case_a", "condition": "base_repair", "attempt": 2, "success": False},
            {"case_id": "case_b", "condition": "base_repair", "attempt": 3, "success": False},
            {"case_id": "case_c", "condition": "cot_repair", "attempt": 1, "success": False},
        ],
    )

    exit_code = extract_failed_cases.main(
        [
            str(input_path),
            "--condition",
            "base_repair",
            "--max-attempt",
            "2",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_a\n"
