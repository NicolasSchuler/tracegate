"""Typed contracts for SWE-Bench Pro generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# Reuse condition constants from the Lite module

SWEBENCH_PRO_DATASET_NAME = "ScaleAI/SWE-bench_Pro"
SWEBENCH_PRO_DEFAULT_SPLIT = "test"

# Pre-built Docker images on DockerHub
SWEBENCH_PRO_DOCKERHUB_REPO = "jefzda/sweap-images"

# SWE-bench Pro GitHub repository for run_scripts and Dockerfiles
SWEBENCH_PRO_GITHUB_REPO = "scaleapi/SWE-bench_Pro-os"


class ProInstanceRow(TypedDict, total=False):
    """Schema for a single SWE-bench Pro dataset row.

    All fields are optional to allow partial loading; the dataset itself
    guarantees ``instance_id`` is always present.
    """

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    requirements: str
    interface: str
    repo_language: str
    fail_to_pass: str
    pass_to_pass: str
    issue_specificity: str
    issue_categories: str
    before_repo_set_cmd: str
    selected_test_files_to_run: str
    dockerhub_tag: str


@dataclass(frozen=True)
class ProPredictionRecord:
    """One SWE-Bench Pro prediction line (patch format for the eval script)."""

    instance_id: str
    patch: str
    prefix: str

    def as_json(self) -> dict[str, str]:
        """Serialise to the flat dict format expected by swe_bench_pro_eval.py."""
        return {
            "instance_id": self.instance_id,
            "patch": self.patch,
            "prefix": self.prefix,
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
class ScoreInstanceResult:
    """Per-instance scoring result."""

    instance_id: str
    passed: bool
    fail_to_pass_met: bool
    pass_to_pass_met: bool
    error: str | None = None
    infrastructure_error: bool = False


@dataclass(frozen=True)
class ScoreConditionSummary:
    """Docker scoring outcome for one condition."""

    condition: str
    run_id: str
    predictions_path: Path
    output_dir: Path
    total_instances: int
    passed_count: int
    failed_count: int
    error_count: int
    accuracy: float


@dataclass(frozen=True)
class ScoreRunSummary:
    """Docker scoring outcomes across conditions."""

    run_dir: Path
    condition_summaries: tuple[ScoreConditionSummary, ...]


@dataclass(frozen=True)
class ScoringSmokeSubsetSummary:
    """Prepared subset run for scoring smoke tests."""

    source_run_dir: Path
    output_run_dir: Path
    instance_count: int
    conditions: tuple[str, ...]
