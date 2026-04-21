"""Typed contracts for SWE-Bench Lite generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONDITION_BASELINE = "baseline"
CONDITION_LLMDEBUG = "llmdebug"
VALID_CONDITIONS = (CONDITION_BASELINE, CONDITION_LLMDEBUG)

SWEBENCH_LITE_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
SWEBENCH_LITE_DEFAULT_SPLIT = "test"


@dataclass(frozen=True)
class PredictionRecord:
    """One official SWE-Bench prediction line."""

    instance_id: str
    model_patch: str
    model_name_or_path: str

    def as_json(self) -> dict[str, str]:
        """Serialise to the flat dict format expected by the SWE-Bench evaluation harness.

        The returned mapping (``instance_id``, ``model_patch``,
        ``model_name_or_path``) is written as one JSON line per prediction in
        the predictions file consumed by ``swebench.harness.run_evaluation``.
        """
        return {
            "instance_id": self.instance_id,
            "model_patch": self.model_patch,
            "model_name_or_path": self.model_name_or_path,
        }


@dataclass(frozen=True)
class ConditionGenerationSummary:
    """Generation stats for one condition."""

    condition: str
    predictions_path: Path
    total_instances: int
    generated: int
    resumed: int
    empty_patch_count: int
    request_error_count: int


@dataclass(frozen=True)
class GenerationRunSummary:
    """Generation stats for an entire run."""

    run_id: str
    run_dir: Path
    instance_count: int
    condition_summaries: tuple[ConditionGenerationSummary, ...]


@dataclass(frozen=True)
class ScoreConditionSummary:
    """Official scoring outcome for one condition."""

    condition: str
    run_id: str
    predictions_path: Path
    working_dir: Path
    log_path: Path
    command: tuple[str, ...]
    exit_code: int


@dataclass(frozen=True)
class ScoreRunSummary:
    """Official scoring outcomes across conditions."""

    run_dir: Path
    condition_summaries: tuple[ScoreConditionSummary, ...]


@dataclass(frozen=True)
class ScoringSmokeSubsetSummary:
    """Prepared subset run for official scoring smoke tests."""

    source_run_dir: Path
    output_run_dir: Path
    instance_count: int
    conditions: tuple[str, ...]
