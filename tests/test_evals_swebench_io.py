"""Tests for SWE-bench I/O utilities (paths, predictions, paired validation)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evals.swebench.io import (
    copy_instructions_path,
    ensure_run_layout,
    load_instance_ids,
    parse_paired_conditions,
    prediction_path,
    read_predictions,
    run_index_path,
    validate_paired_predictions,
    validate_prediction_file,
    write_copy_instructions,
    write_instance_ids,
    write_predictions,
    write_run_index,
)
from evals.swebench.types import PredictionRecord
from llmdebug import _json as json


def _write_records(path: Path, ids: list[str], *, model_tag: str) -> None:
    records = [
        PredictionRecord(
            instance_id=instance_id,
            model_patch=f"diff --git a/f.py b/f.py\n+{instance_id}\n",
            model_name_or_path=model_tag,
        )
        for instance_id in ids
    ]
    write_predictions(path, records)


def test_validate_prediction_file_reports_duplicate_instance_ids(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_predictions(
        path,
        [
            PredictionRecord("inst-1", "diff --git a/a.py b/a.py\n", "m"),
            PredictionRecord("inst-1", "diff --git a/a.py b/a.py\n", "m"),
        ],
    )

    rows, duplicates = validate_prediction_file(path)

    assert rows == 2
    assert duplicates == 1


def test_validate_paired_predictions_requires_all_condition_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    _write_records(
        prediction_path(run_dir, "baseline"),
        ["inst-1", "inst-2"],
        model_tag="model::baseline",
    )

    with pytest.raises(ValueError, match="missing predictions file"):
        validate_paired_predictions(
            run_dir,
            conditions=["baseline", "llmdebug"],
            require_files=True,
        )


def test_validate_paired_predictions_rejects_condition_skew(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_records(
        prediction_path(run_dir, "baseline"),
        ["inst-1", "inst-2"],
        model_tag="model::baseline",
    )
    _write_records(
        prediction_path(run_dir, "llmdebug"),
        ["inst-1", "inst-3"],
        model_tag="model::llmdebug",
    )

    with pytest.raises(ValueError, match="paired prediction mismatch"):
        validate_paired_predictions(run_dir, conditions=["baseline", "llmdebug"])


def test_parse_paired_conditions_rejects_single_condition() -> None:
    with pytest.raises(ValueError, match="conditions must include exactly baseline,llmdebug"):
        parse_paired_conditions("baseline")


def test_validate_paired_predictions_accepts_matching_dual_condition_sets(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    ids = ["inst-1", "inst-2", "inst-3"]
    _write_records(
        prediction_path(run_dir, "baseline"),
        ids,
        model_tag="model::baseline",
    )
    _write_records(
        prediction_path(run_dir, "llmdebug"),
        ids,
        model_tag="model::llmdebug",
    )

    # Should not raise — matching conditions validate successfully
    result = validate_paired_predictions(
        run_dir,
        conditions=["baseline", "llmdebug"],
        expected_ids=ids,
        require_files=True,
    )
    assert result is None


def test_write_handoff_index_and_copy_instructions(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_records(prediction_path(run_dir, "baseline"), ["inst-1"], model_tag="baseline-model")
    _write_records(prediction_path(run_dir, "llmdebug"), ["inst-1"], model_tag="llmdebug-model")
    write_instance_ids(run_dir / "instance_ids.txt", ["inst-1"])
    (run_dir / "run_metadata.json").write_text('{"run_id":"demo"}\n', encoding="utf-8")

    index = write_run_index(run_dir, run_id="demo", conditions=["baseline", "llmdebug"])
    instructions = write_copy_instructions(run_dir, conditions=["baseline", "llmdebug"])

    assert index == run_index_path(run_dir)
    assert instructions == copy_instructions_path(run_dir)

    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["run_id"] == "demo"
    assert "baseline" in payload["predictions"]
    assert "llmdebug" in payload["predictions"]
    assert "<copied_run_dir>" in instructions.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# read_predictions
# ---------------------------------------------------------------------------


def test_read_predictions_parses_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    lines = [
        json.dumps({"instance_id": "a/1", "model_patch": "diff1", "model_name_or_path": "m1"}),
        json.dumps({"instance_id": "b/2", "model_patch": "diff2", "model_name_or_path": "m2"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = read_predictions(path)

    assert len(records) == 2
    assert records[0] == PredictionRecord("a/1", "diff1", "m1")
    assert records[1] == PredictionRecord("b/2", "diff2", "m2")


def test_read_predictions_returns_empty_list_for_nonexistent_file(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.jsonl"
    assert read_predictions(path) == []


def test_read_predictions_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    line = json.dumps({"instance_id": "x", "model_patch": "p", "model_name_or_path": "m"})
    path.write_text(f"\n  \n{line}\n\n{line}\n  \n", encoding="utf-8")

    records = read_predictions(path)

    assert len(records) == 2


def test_read_predictions_raises_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    good = json.dumps({"instance_id": "x", "model_patch": "p", "model_name_or_path": "m"})
    path.write_text(f"{good}\n{{not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}:2:.*invalid JSON"):
        read_predictions(path)


def test_read_predictions_raises_on_missing_instance_id(tmp_path: Path) -> None:
    path = tmp_path / "no_id.jsonl"
    line = json.dumps({"model_patch": "p", "model_name_or_path": "m"})
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing instance_id"):
        read_predictions(path)


def test_read_predictions_raises_on_non_string_model_patch(tmp_path: Path) -> None:
    path = tmp_path / "bad_patch.jsonl"
    line = json.dumps({"instance_id": "x", "model_patch": 123, "model_name_or_path": "m"})
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_patch must be string"):
        read_predictions(path)


def test_read_predictions_raises_on_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_predictions(path) == []


# ---------------------------------------------------------------------------
# load_instance_ids
# ---------------------------------------------------------------------------


def test_load_instance_ids_reads_lines(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    ids = load_instance_ids(path)

    assert ids == ["alpha", "beta", "gamma"]


def test_load_instance_ids_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("\nalpha\n  \nbeta\n\n", encoding="utf-8")

    ids = load_instance_ids(path)

    assert ids == ["alpha", "beta"]


def test_load_instance_ids_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("first\nsecond\nfirst\nthird\nsecond\n", encoding="utf-8")

    ids = load_instance_ids(path)

    assert ids == ["first", "second", "third"]


def test_load_instance_ids_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("", encoding="utf-8")

    assert load_instance_ids(path) == []


# ---------------------------------------------------------------------------
# ensure_run_layout
# ---------------------------------------------------------------------------


def test_ensure_run_layout_creates_directories(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "my_run"

    ensure_run_layout(run_dir, conditions=["baseline", "llmdebug"])

    assert run_dir.is_dir()
    assert (run_dir / "baseline").is_dir()
    assert (run_dir / "llmdebug").is_dir()


def test_ensure_run_layout_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    ensure_run_layout(run_dir, conditions=["baseline", "llmdebug"])
    ensure_run_layout(run_dir, conditions=["baseline", "llmdebug"])

    assert run_dir.is_dir()
    assert (run_dir / "baseline").is_dir()
    assert (run_dir / "llmdebug").is_dir()


def test_ensure_run_layout_rejects_invalid_condition(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="unsupported condition"):
        ensure_run_layout(run_dir, conditions=["baseline", "INVALID"])


# ---------------------------------------------------------------------------
# repo_root (line 24)
# ---------------------------------------------------------------------------


def test_repo_root_returns_path() -> None:
    from evals.swebench.io import repo_root

    result = repo_root()
    assert isinstance(result, Path)
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# validate_run_id (lines 58, 60)
# ---------------------------------------------------------------------------


def test_validate_run_id_rejects_empty_string() -> None:
    from evals.swebench.io import validate_run_id

    with pytest.raises(ValueError, match="run_id must be non-empty"):
        validate_run_id("   ")


def test_validate_run_id_rejects_bad_characters() -> None:
    from evals.swebench.io import validate_run_id

    with pytest.raises(ValueError, match="unsupported characters"):
        validate_run_id("run id with spaces")


# ---------------------------------------------------------------------------
# parse_conditions (lines 102, 108, 114, 118)
# ---------------------------------------------------------------------------


def test_parse_conditions_none_returns_all() -> None:
    from evals.swebench.io import parse_conditions

    result = parse_conditions(None)
    assert set(result) == {"baseline", "llmdebug"}


def test_parse_conditions_empty_string_returns_all() -> None:
    from evals.swebench.io import parse_conditions

    result = parse_conditions(",  , ")
    assert set(result) == {"baseline", "llmdebug"}


def test_parse_conditions_rejects_invalid_token() -> None:
    from evals.swebench.io import parse_conditions

    with pytest.raises(ValueError, match="unsupported condition"):
        parse_conditions("baseline,bogus")


def test_parse_conditions_deduplicates() -> None:
    from evals.swebench.io import parse_conditions

    result = parse_conditions("baseline,baseline,llmdebug")
    assert result == ["baseline", "llmdebug"]


# ---------------------------------------------------------------------------
# load_json — non-dict JSON (lines 262-265)
# ---------------------------------------------------------------------------


def test_load_json_rejects_non_dict(tmp_path: Path) -> None:
    from evals.swebench.io import load_json

    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(path)


# ---------------------------------------------------------------------------
# write_instance_ids — blank/duplicate skipping (line 310)
# ---------------------------------------------------------------------------


def test_write_instance_ids_skips_blank_and_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    write_instance_ids(path, ["a", "", "  ", "b", "a", "c"])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# read_predictions — non-dict payload (line 347)
# ---------------------------------------------------------------------------


def test_read_predictions_rejects_array_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction row must be object"):
        read_predictions(path)


# ---------------------------------------------------------------------------
# read_predictions — missing model_name_or_path (line 357)
# ---------------------------------------------------------------------------


def test_read_predictions_rejects_missing_model_name(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    line = json.dumps({"instance_id": "x", "model_patch": "p", "model_name_or_path": ""})
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing model_name_or_path"):
        read_predictions(path)


# ---------------------------------------------------------------------------
# write_predictions — finally cleanup temp file (line 401)
# ---------------------------------------------------------------------------


def test_write_predictions_cleans_temp_on_error(tmp_path: Path) -> None:
    """When an exception is raised during write, temp file is cleaned up."""
    path = tmp_path / "preds.jsonl"

    def _bad_records():
        yield PredictionRecord("a", "diff", "m")
        raise RuntimeError("boom during serialization")

    with pytest.raises(RuntimeError, match="boom during serialization"):
        from evals.swebench.io import write_predictions as _wp

        _wp(path, _bad_records())

    # The temp file should have been cleaned up; only check nothing is left
    leftover = list(tmp_path.glob(".*predictions*.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# collect_prediction_ids (lines 414-415)
# ---------------------------------------------------------------------------


def test_collect_prediction_ids_returns_ordered_ids(tmp_path: Path) -> None:
    from evals.swebench.io import collect_prediction_ids

    path = tmp_path / "preds.jsonl"
    write_predictions(
        path,
        [
            PredictionRecord("b", "diff", "m"),
            PredictionRecord("a", "diff", "m"),
        ],
    )
    ids = collect_prediction_ids(path)
    assert ids == ["b", "a"]


# ---------------------------------------------------------------------------
# validate_paired_predictions edge cases (lines 481, 485, 501-503)
# ---------------------------------------------------------------------------


def test_validate_paired_predictions_rejects_empty_predictions(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    # Write empty files
    (run_dir / "baseline" / "predictions.jsonl").write_text("", encoding="utf-8")
    (run_dir / "llmdebug" / "predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="predictions file is empty"):
        validate_paired_predictions(
            run_dir,
            conditions=["baseline", "llmdebug"],
            require_files=True,
            require_nonempty=True,
        )


def test_validate_paired_predictions_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    # Manually write duplicates
    line = json.dumps({"instance_id": "x", "model_patch": "p", "model_name_or_path": "m"})
    (run_dir / "baseline" / "predictions.jsonl").write_text(f"{line}\n{line}\n", encoding="utf-8")
    _write_records(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["x"],
        model_tag="m",
    )
    with pytest.raises(ValueError, match="duplicate instance_id"):
        validate_paired_predictions(
            run_dir,
            conditions=["baseline", "llmdebug"],
        )


def test_validate_paired_predictions_rejects_expected_ids_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_records(
        run_dir / "baseline" / "predictions.jsonl",
        ["x"],
        model_tag="m",
    )
    _write_records(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["x"],
        model_tag="m",
    )
    with pytest.raises(ValueError, match="prediction instance set differs"):
        validate_paired_predictions(
            run_dir,
            conditions=["baseline", "llmdebug"],
            expected_ids=["x", "y"],
        )
