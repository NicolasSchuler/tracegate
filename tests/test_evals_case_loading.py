"""Tests for eval case discovery, loading, and metadata filtering."""

from __future__ import annotations

import pytest

from evals.run_eval import load_cases
from llmdebug import _json as json


def _write_case(tmp_path, case_id: str, metadata: dict) -> None:
    case_dir = tmp_path / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(metadata), encoding="utf-8")
    (case_dir / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")


def test_case_loading_valid_and_defaults(tmp_path) -> None:
    _write_case(
        tmp_path,
        "valid_case",
        {
            "category": "hidden_state",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
        },
    )

    cases = load_cases(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "valid_case"
    assert case.category == "hidden_state"
    assert case.difficulty == "easy"
    assert case.snapshot_dependency == "required"
    assert case.expected_exception == "AssertionError"
    assert case.tags == []
    assert case.python_args == ["-m", "pytest", "-q"]
    assert case.timeout_sec == 120
    assert case.description is None


def test_case_loading_rejects_invalid_category(tmp_path) -> None:
    _write_case(
        tmp_path,
        "invalid_category",
        {
            "category": "bad_category",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
        },
    )

    with pytest.raises(ValueError, match="category must be one of"):
        load_cases(tmp_path)


def test_case_loading_rejects_invalid_difficulty(tmp_path) -> None:
    _write_case(
        tmp_path,
        "invalid_difficulty",
        {
            "category": "hidden_state",
            "difficulty": "tricky",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
        },
    )

    with pytest.raises(ValueError, match="difficulty must be one of"):
        load_cases(tmp_path)


def test_case_loading_rejects_invalid_snapshot_dependency(tmp_path) -> None:
    _write_case(
        tmp_path,
        "invalid_snapshot_dependency",
        {
            "category": "hidden_state",
            "difficulty": "easy",
            "snapshot_dependency": "maybe",
            "expected_exception": "AssertionError",
        },
    )

    with pytest.raises(ValueError, match="snapshot_dependency must be one of"):
        load_cases(tmp_path)


def test_case_loading_rejects_invalid_tags(tmp_path) -> None:
    _write_case(
        tmp_path,
        "invalid_tags",
        {
            "category": "hidden_state",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
            "tags": ["ok", ""],
        },
    )

    with pytest.raises(ValueError, match="tags must be an array of non-empty strings"):
        load_cases(tmp_path)
