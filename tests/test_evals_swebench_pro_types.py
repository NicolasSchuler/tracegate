"""Tests for SWE-bench Pro typed contracts (constants, dataclasses, TypedDict)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from evals.swebench_pro.types import (
    SWEBENCH_PRO_DATASET_NAME,
    SWEBENCH_PRO_DEFAULT_SPLIT,
    SWEBENCH_PRO_DOCKERHUB_REPO,
    SWEBENCH_PRO_GITHUB_REPO,
    ConditionGenerationSummary,
    GenerationRunSummary,
    ScoreConditionSummary,
    ScoreInstanceResult,
    ScoreRunSummary,
    ScoringSmokeSubsetSummary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_dataset_name() -> None:
    assert SWEBENCH_PRO_DATASET_NAME == "ScaleAI/SWE-bench_Pro"


def test_default_split() -> None:
    assert SWEBENCH_PRO_DEFAULT_SPLIT == "test"


def test_dockerhub_repo() -> None:
    assert SWEBENCH_PRO_DOCKERHUB_REPO == "jefzda/sweap-images"


def test_github_repo() -> None:
    assert SWEBENCH_PRO_GITHUB_REPO == "scaleapi/SWE-bench_Pro-os"


# ---------------------------------------------------------------------------
# ConditionGenerationSummary
# ---------------------------------------------------------------------------


def test_condition_generation_summary_is_frozen() -> None:
    summary = ConditionGenerationSummary(
        condition="baseline",
        predictions_path=Path("/tmp/preds.jsonl"),
        total_instances=100,
        generated=90,
        resumed=10,
        empty_patch_count=5,
        request_error_count=2,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.condition = "changed"  # type: ignore[misc]


def test_condition_generation_summary_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(ConditionGenerationSummary)]
    assert field_names == [
        "condition",
        "predictions_path",
        "total_instances",
        "generated",
        "resumed",
        "empty_patch_count",
        "request_error_count",
    ]


# ---------------------------------------------------------------------------
# GenerationRunSummary
# ---------------------------------------------------------------------------


def test_generation_run_summary_is_frozen() -> None:
    summary = GenerationRunSummary(
        run_id="run-01",
        run_dir=Path("/tmp/run-01"),
        instance_count=50,
        condition_summaries=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.run_id = "changed"  # type: ignore[misc]


def test_generation_run_summary_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(GenerationRunSummary)]
    assert field_names == ["run_id", "run_dir", "instance_count", "condition_summaries"]


def test_generation_run_summary_condition_summaries_is_tuple() -> None:
    cond = ConditionGenerationSummary(
        condition="baseline",
        predictions_path=Path("/tmp/p.jsonl"),
        total_instances=10,
        generated=8,
        resumed=2,
        empty_patch_count=0,
        request_error_count=0,
    )
    summary = GenerationRunSummary(
        run_id="run-01",
        run_dir=Path("/tmp/run-01"),
        instance_count=10,
        condition_summaries=(cond,),
    )
    assert isinstance(summary.condition_summaries, tuple)
    assert len(summary.condition_summaries) == 1
    assert summary.condition_summaries[0].condition == "baseline"


# ---------------------------------------------------------------------------
# ScoreInstanceResult
# ---------------------------------------------------------------------------


def test_score_instance_result_is_frozen() -> None:
    result = ScoreInstanceResult(
        instance_id="x",
        passed=True,
        fail_to_pass_met=True,
        pass_to_pass_met=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.passed = False  # type: ignore[misc]


def test_score_instance_result_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(ScoreInstanceResult)]
    assert field_names == [
        "instance_id",
        "passed",
        "fail_to_pass_met",
        "pass_to_pass_met",
        "error",
        "infrastructure_error",
    ]


def test_score_instance_result_error_defaults_to_none() -> None:
    result = ScoreInstanceResult(
        instance_id="a/1",
        passed=False,
        fail_to_pass_met=False,
        pass_to_pass_met=True,
    )
    assert result.error is None
    assert result.infrastructure_error is False


def test_score_instance_result_error_can_be_set() -> None:
    result = ScoreInstanceResult(
        instance_id="a/1",
        passed=False,
        fail_to_pass_met=False,
        pass_to_pass_met=False,
        error="container timed out",
    )
    assert result.error == "container timed out"


def test_score_instance_result_infrastructure_error_can_be_set() -> None:
    result = ScoreInstanceResult(
        instance_id="a/1",
        passed=False,
        fail_to_pass_met=False,
        pass_to_pass_met=False,
        error="docker SDK not installed",
        infrastructure_error=True,
    )
    assert result.infrastructure_error is True


# ---------------------------------------------------------------------------
# ScoreConditionSummary
# ---------------------------------------------------------------------------


def test_score_condition_summary_is_frozen() -> None:
    summary = ScoreConditionSummary(
        condition="llmdebug",
        run_id="run-01",
        predictions_path=Path("/tmp/preds.jsonl"),
        output_dir=Path("/tmp/output"),
        total_instances=50,
        passed_count=40,
        failed_count=8,
        error_count=2,
        accuracy=0.80,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.condition = "changed"  # type: ignore[misc]


def test_score_condition_summary_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(ScoreConditionSummary)]
    assert field_names == [
        "condition",
        "run_id",
        "predictions_path",
        "output_dir",
        "total_instances",
        "passed_count",
        "failed_count",
        "error_count",
        "accuracy",
    ]


# ---------------------------------------------------------------------------
# ScoreRunSummary
# ---------------------------------------------------------------------------


def test_score_run_summary_is_frozen() -> None:
    summary = ScoreRunSummary(
        run_dir=Path("/tmp/run"),
        condition_summaries=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.run_dir = Path("/other")  # type: ignore[misc]


def test_score_run_summary_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(ScoreRunSummary)]
    assert field_names == ["run_dir", "condition_summaries"]


# ---------------------------------------------------------------------------
# ScoringSmokeSubsetSummary
# ---------------------------------------------------------------------------


def test_scoring_smoke_subset_summary_is_frozen() -> None:
    summary = ScoringSmokeSubsetSummary(
        source_run_dir=Path("/tmp/source"),
        output_run_dir=Path("/tmp/output"),
        instance_count=5,
        conditions=("baseline", "llmdebug"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.instance_count = 10  # type: ignore[misc]


def test_scoring_smoke_subset_summary_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(ScoringSmokeSubsetSummary)]
    assert field_names == [
        "source_run_dir",
        "output_run_dir",
        "instance_count",
        "conditions",
    ]


# ---------------------------------------------------------------------------
# __init__.py public API re-exports
# ---------------------------------------------------------------------------


def test_init_exports_prediction_record() -> None:
    from evals.swebench_pro import PredictionRecord

    assert PredictionRecord is not None
    record = PredictionRecord(instance_id="x", model_patch="p", model_name_or_path="m")
    assert record.instance_id == "x"


def test_init_exports_condition_constants() -> None:
    from evals.swebench_pro import CONDITION_BASELINE, CONDITION_LLMDEBUG

    assert CONDITION_BASELINE == "baseline"
    assert CONDITION_LLMDEBUG == "llmdebug"


def test_init_all_contains_expected_symbols() -> None:
    from evals.swebench_pro import __all__

    expected = {
        "CONDITION_BASELINE",
        "CONDITION_LLMDEBUG",
        "SWEBENCH_PRO_DATASET_NAME",
        "ConditionGenerationSummary",
        "GenerationRunSummary",
        "PredictionRecord",
        "ScoreRunSummary",
        "ScoringSmokeSubsetSummary",
    }
    assert set(__all__) == expected


def test_init_exports_are_importable() -> None:
    """Every name listed in __all__ must be importable from the package."""
    import evals.swebench_pro as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), f"{name!r} listed in __all__ but not importable"
