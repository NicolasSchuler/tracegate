"""Tests for SWE-bench scoring smoke-test subset preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.swebench.io import write_instance_ids, write_predictions
from evals.swebench.smoke import prepare_scoring_smoke_subset
from evals.swebench.types import PredictionRecord


def _records(ids: list[str], model_tag: str) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            instance_id=item,
            model_patch="diff --git a/a.py b/a.py\n",
            model_name_or_path=model_tag,
        )
        for item in ids
    ]


def test_prepare_scoring_smoke_subset_writes_paired_subset(tmp_path) -> None:
    source = tmp_path / "source-run"
    (source / "baseline").mkdir(parents=True, exist_ok=True)
    (source / "llmdebug").mkdir(parents=True, exist_ok=True)
    ids = ["inst-1", "inst-2", "inst-3"]
    write_predictions(source / "baseline" / "predictions.jsonl", _records(ids, "baseline-model"))
    write_predictions(source / "llmdebug" / "predictions.jsonl", _records(ids, "llmdebug-model"))
    write_instance_ids(source / "instance_ids.txt", ids)

    output = tmp_path / "smoke-run"
    summary = prepare_scoring_smoke_subset(
        source_run_dir=source,
        output_run_dir=output,
        max_instances=2,
        conditions="baseline,llmdebug",
    )

    assert summary.instance_count == 2
    assert (output / "baseline" / "predictions.jsonl").exists()
    assert (output / "llmdebug" / "predictions.jsonl").exists()
    assert (output / "COPY_INSTRUCTIONS.txt").exists()
    assert (output / "run_index.json").exists()
    assert (output / "run_metadata.json").exists()
    assert (output / "instance_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "inst-1",
        "inst-2",
    ]


def test_prepare_scoring_smoke_subset_fails_when_output_not_empty(tmp_path) -> None:
    source = tmp_path / "source-run"
    (source / "baseline").mkdir(parents=True, exist_ok=True)
    (source / "llmdebug").mkdir(parents=True, exist_ok=True)
    ids = ["inst-1", "inst-2"]
    write_predictions(source / "baseline" / "predictions.jsonl", _records(ids, "baseline-model"))
    write_predictions(source / "llmdebug" / "predictions.jsonl", _records(ids, "llmdebug-model"))
    write_instance_ids(source / "instance_ids.txt", ids)

    output = tmp_path / "smoke-run"
    output.mkdir(parents=True, exist_ok=True)
    (output / "keep.txt").write_text("x", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists and is not empty"):
        prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=1,
            conditions="baseline,llmdebug",
        )


# ---------------------------------------------------------------------------
# Validation edge cases (lines 34, 36, 54, 69)
# ---------------------------------------------------------------------------


def test_prepare_smoke_rejects_max_instances_below_one(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_instances must be >= 1"):
        prepare_scoring_smoke_subset(
            source_run_dir=tmp_path,
            output_run_dir=tmp_path / "out",
            max_instances=0,
            conditions="baseline,llmdebug",
        )


def test_prepare_smoke_rejects_missing_source_dir(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="source run directory does not exist"):
        prepare_scoring_smoke_subset(
            source_run_dir=tmp_path / "nonexistent",
            output_run_dir=tmp_path / "out",
            max_instances=1,
            conditions="baseline,llmdebug",
        )


def test_prepare_smoke_rejects_empty_source_ids(tmp_path) -> None:
    source = tmp_path / "source-run"
    (source / "baseline").mkdir(parents=True, exist_ok=True)
    (source / "llmdebug").mkdir(parents=True, exist_ok=True)
    # Write valid but empty predictions (validate step passes when
    # we monkeypatch it, but source_ids ends up empty)
    write_predictions(source / "baseline" / "predictions.jsonl", [])
    write_predictions(source / "llmdebug" / "predictions.jsonl", [])
    # Write an empty instance_ids.txt → no IDs
    write_instance_ids(source / "instance_ids.txt", [])

    # Monkeypatch validate to pass (empty predictions would fail it)
    import evals.swebench.smoke as smoke_mod

    original_validate = smoke_mod.validate_paired_predictions

    def _noop_validate(*_a, **_kw):
        pass

    smoke_mod.validate_paired_predictions = _noop_validate
    try:
        with pytest.raises(RuntimeError, match="source run has no instance ids"):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=tmp_path / "out",
                max_instances=1,
                conditions="baseline,llmdebug",
            )
    finally:
        smoke_mod.validate_paired_predictions = original_validate


def test_prepare_smoke_raises_on_subset_length_mismatch(tmp_path) -> None:
    """Condition missing a selected ID → RuntimeError (line 69)."""
    source = tmp_path / "source-run"
    (source / "baseline").mkdir(parents=True, exist_ok=True)
    (source / "llmdebug").mkdir(parents=True, exist_ok=True)
    ids = ["inst-1", "inst-2"]
    write_predictions(
        source / "baseline" / "predictions.jsonl",
        _records(ids, "baseline-model"),
    )
    # llmdebug only has inst-1 (missing inst-2)
    write_predictions(
        source / "llmdebug" / "predictions.jsonl",
        _records(["inst-1"], "llmdebug-model"),
    )
    write_instance_ids(source / "instance_ids.txt", ids)

    import evals.swebench.smoke as smoke_mod

    original_validate = smoke_mod.validate_paired_predictions

    def _noop_validate(*_a, **_kw):
        pass

    smoke_mod.validate_paired_predictions = _noop_validate
    try:
        with pytest.raises(
            RuntimeError,
            match="does not contain all selected subset ids",
        ):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=tmp_path / "out",
                max_instances=2,
                conditions="baseline,llmdebug",
            )
    finally:
        smoke_mod.validate_paired_predictions = original_validate


# ---------------------------------------------------------------------------
# _source_instance_order fallback (line 107)
# ---------------------------------------------------------------------------


def test_source_instance_order_falls_back_to_predictions(
    tmp_path,
) -> None:
    """When instance_ids.txt is empty, falls back to predictions."""
    from evals.swebench.smoke import _source_instance_order

    source = tmp_path / "source-run"
    (source / "baseline").mkdir(parents=True, exist_ok=True)
    # Write empty instance_ids.txt
    write_instance_ids(source / "instance_ids.txt", [])
    # Write predictions with IDs
    write_predictions(
        source / "baseline" / "predictions.jsonl",
        _records(["inst-a", "inst-b"], "m"),
    )

    result = _source_instance_order(source_run_dir=source, fallback_condition="baseline")
    assert result == ["inst-a", "inst-b"]
