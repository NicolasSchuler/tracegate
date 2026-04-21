"""Tests for SWE-bench Pro I/O utilities (paths, predictions, paired validation)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evals.swebench.types import PredictionRecord
from evals.swebench_pro.io import (
    collect_prediction_ids,
    copy_instructions_path,
    ensure_run_layout,
    instance_ids_path,
    load_instance_ids,
    load_json,
    metadata_path,
    parse_conditions,
    parse_paired_conditions,
    prediction_path,
    read_predictions,
    run_dir_for_id,
    run_index_path,
    swebench_pro_runs_root,
    validate_paired_predictions,
    validate_prediction_file,
    validate_run_id,
    write_copy_instructions,
    write_instance_ids,
    write_json,
    write_predictions,
    write_run_index,
)
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


# ---------------------------------------------------------------------------
# swebench_pro_runs_root
# ---------------------------------------------------------------------------


def test_swebench_pro_runs_root_ends_with_expected_suffix() -> None:
    root = swebench_pro_runs_root()
    assert root.parts[-2:] == ("evals", "swebench_pro_runs")


def test_swebench_pro_runs_root_respects_override(tmp_path: Path) -> None:
    root = swebench_pro_runs_root(root=tmp_path)
    assert root == tmp_path / "evals" / "swebench_pro_runs"


# ---------------------------------------------------------------------------
# validate_run_id
# ---------------------------------------------------------------------------


def test_validate_run_id_accepts_valid_ids() -> None:
    assert validate_run_id("my-run-01") == "my-run-01"
    assert validate_run_id("run_v2.3") == "run_v2.3"
    assert validate_run_id("RUN-123") == "RUN-123"
    assert validate_run_id("simple") == "simple"


def test_validate_run_id_strips_whitespace() -> None:
    assert validate_run_id("  my-run  ") == "my-run"


def test_validate_run_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_run_id("")


def test_validate_run_id_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_run_id("   ")


def test_validate_run_id_rejects_slashes() -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        validate_run_id("my/run")


def test_validate_run_id_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        validate_run_id("../escape")


def test_validate_run_id_rejects_spaces_inside() -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        validate_run_id("my run")


# ---------------------------------------------------------------------------
# run_dir_for_id
# ---------------------------------------------------------------------------


def test_run_dir_for_id_returns_correct_path(tmp_path: Path) -> None:
    result = run_dir_for_id("my-run", root=tmp_path)
    expected = tmp_path / "evals" / "swebench_pro_runs" / "my-run"
    assert result == expected


def test_run_dir_for_id_validates_run_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_dir_for_id("")


# ---------------------------------------------------------------------------
# prediction_path / metadata_path / instance_ids_path
# ---------------------------------------------------------------------------


def test_prediction_path_returns_correct_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert prediction_path(run_dir, "baseline") == run_dir / "baseline" / "predictions.jsonl"
    assert prediction_path(run_dir, "llmdebug") == run_dir / "llmdebug" / "predictions.jsonl"


def test_prediction_path_rejects_invalid_condition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported condition"):
        prediction_path(tmp_path / "run", "INVALID")


def test_metadata_path_returns_correct_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert metadata_path(run_dir) == run_dir / "run_metadata.json"


def test_instance_ids_path_returns_correct_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert instance_ids_path(run_dir) == run_dir / "instance_ids.txt"


def test_copy_instructions_path_returns_correct_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert copy_instructions_path(run_dir) == run_dir / "COPY_INSTRUCTIONS.txt"


def test_run_index_path_returns_correct_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert run_index_path(run_dir) == run_dir / "run_index.json"


# ---------------------------------------------------------------------------
# parse_conditions / parse_paired_conditions
# ---------------------------------------------------------------------------


def test_parse_conditions_returns_all_for_none() -> None:
    result = parse_conditions(None)
    assert result == ["baseline", "llmdebug"]


def test_parse_conditions_parses_comma_separated_string() -> None:
    result = parse_conditions("baseline,llmdebug")
    assert result == ["baseline", "llmdebug"]


def test_parse_conditions_accepts_single_condition() -> None:
    result = parse_conditions("baseline")
    assert result == ["baseline"]


def test_parse_conditions_rejects_invalid_condition() -> None:
    with pytest.raises(ValueError, match="unsupported condition"):
        parse_conditions("baseline,INVALID")


def test_parse_conditions_deduplicates() -> None:
    result = parse_conditions("baseline,baseline,llmdebug")
    assert result == ["baseline", "llmdebug"]


def test_parse_conditions_returns_all_for_empty_string() -> None:
    result = parse_conditions("")
    assert result == ["baseline", "llmdebug"]


def test_parse_paired_conditions_accepts_all() -> None:
    result = parse_paired_conditions("baseline,llmdebug")
    assert result == ["baseline", "llmdebug"]


def test_parse_paired_conditions_accepts_none() -> None:
    result = parse_paired_conditions(None)
    assert result == ["baseline", "llmdebug"]


def test_parse_paired_conditions_rejects_single_condition() -> None:
    with pytest.raises(ValueError, match="conditions must include exactly baseline,llmdebug"):
        parse_paired_conditions("baseline")


def test_parse_paired_conditions_rejects_single_llmdebug() -> None:
    with pytest.raises(ValueError, match="conditions must include exactly baseline,llmdebug"):
        parse_paired_conditions("llmdebug")


# ---------------------------------------------------------------------------
# write_predictions / read_predictions round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_predictions_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    records = [
        PredictionRecord("inst-1", "diff --git a/a.py b/a.py\n", "model-a"),
        PredictionRecord("inst-2", "diff --git a/b.py b/b.py\n", "model-a"),
    ]
    write_predictions(path, records)
    loaded = read_predictions(path)

    assert len(loaded) == 2
    assert loaded[0] == records[0]
    assert loaded[1] == records[1]


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


def test_read_predictions_returns_empty_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_predictions(path) == []


# ---------------------------------------------------------------------------
# validate_prediction_file
# ---------------------------------------------------------------------------


def test_validate_prediction_file_reports_duplicate_instance_ids(tmp_path: Path) -> None:
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


def test_validate_prediction_file_no_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_predictions(
        path,
        [
            PredictionRecord("inst-1", "diff1", "m"),
            PredictionRecord("inst-2", "diff2", "m"),
        ],
    )

    rows, duplicates = validate_prediction_file(path)

    assert rows == 2
    assert duplicates == 0


# ---------------------------------------------------------------------------
# collect_prediction_ids
# ---------------------------------------------------------------------------


def test_collect_prediction_ids_returns_ordered_ids(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    _write_records(path, ["c", "a", "b"], model_tag="model")

    ids = collect_prediction_ids(path)

    assert ids == ["c", "a", "b"]


def test_collect_prediction_ids_returns_empty_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    assert collect_prediction_ids(path) == []


# ---------------------------------------------------------------------------
# validate_paired_predictions
# ---------------------------------------------------------------------------


def test_validate_paired_predictions_requires_all_condition_files(tmp_path: Path) -> None:
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


def test_validate_paired_predictions_rejects_condition_skew(tmp_path: Path) -> None:
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


def test_validate_paired_predictions_accepts_matching_dual_condition_sets(
    tmp_path: Path,
) -> None:
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

    validate_paired_predictions(
        run_dir,
        conditions=["baseline", "llmdebug"],
        expected_ids=ids,
        require_files=True,
    )


def test_validate_paired_predictions_rejects_expected_ids_mismatch(
    tmp_path: Path,
) -> None:
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
        ["inst-1", "inst-2"],
        model_tag="model::llmdebug",
    )

    with pytest.raises(ValueError, match="prediction instance set differs"):
        validate_paired_predictions(
            run_dir,
            conditions=["baseline", "llmdebug"],
            expected_ids=["inst-1", "inst-2", "inst-3"],
        )


def test_validate_paired_predictions_rejects_duplicates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)

    # Write a file with duplicate instance_ids manually
    path = prediction_path(run_dir, "baseline")
    write_predictions(
        path,
        [
            PredictionRecord("inst-1", "diff1", "m"),
            PredictionRecord("inst-1", "diff2", "m"),
        ],
    )
    _write_records(
        prediction_path(run_dir, "llmdebug"),
        ["inst-1"],
        model_tag="model::llmdebug",
    )

    with pytest.raises(ValueError, match="duplicate instance_id"):
        validate_paired_predictions(run_dir, conditions=["baseline", "llmdebug"])


# ---------------------------------------------------------------------------
# load_instance_ids / write_instance_ids
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


def test_write_instance_ids_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    original = ["inst-a", "inst-b", "inst-c"]
    write_instance_ids(path, original)
    loaded = load_instance_ids(path)

    assert loaded == original


def test_write_instance_ids_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    write_instance_ids(path, ["a", "b", "a", "c", "b"])
    loaded = load_instance_ids(path)

    assert loaded == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# write_json / load_json
# ---------------------------------------------------------------------------


def test_write_json_and_load_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    payload = {"key": "value", "count": 42}
    write_json(path, payload)
    loaded = load_json(path)

    assert loaded == payload


def test_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "data.json"
    write_json(path, {"hello": "world"})

    assert path.exists()
    loaded = load_json(path)
    assert loaded["hello"] == "world"


def test_load_json_raises_on_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(path)


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
# write_run_index / write_copy_instructions
# ---------------------------------------------------------------------------


def test_write_handoff_index_and_copy_instructions(tmp_path: Path) -> None:
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


def test_write_run_index_contains_schema_version(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_records(prediction_path(run_dir, "baseline"), ["inst-1"], model_tag="m")
    _write_records(prediction_path(run_dir, "llmdebug"), ["inst-1"], model_tag="m")
    write_instance_ids(run_dir / "instance_ids.txt", ["inst-1"])
    (run_dir / "run_metadata.json").write_text('{"run_id":"test"}\n', encoding="utf-8")

    index = write_run_index(run_dir, run_id="test", conditions=["baseline", "llmdebug"])
    payload = json.loads(index.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "swebench_pro_handoff_index_v1"


def test_copy_instructions_mentions_swebench_pro(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)

    instructions = write_copy_instructions(run_dir, conditions=["baseline", "llmdebug"])
    content = instructions.read_text(encoding="utf-8")

    assert "SWE-Bench Pro" in content
    assert "score_swebench_pro_official" in content
    assert "from repo root" in content
    assert str(run_dir / "docker_eval") in content
