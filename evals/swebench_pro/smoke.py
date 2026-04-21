"""Helpers for SWE-Bench Pro scoring smoke subsets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .io import (
    ensure_run_layout,
    instance_ids_path,
    load_instance_ids,
    metadata_path,
    parse_paired_conditions,
    prediction_path,
    read_predictions,
    validate_paired_predictions,
    write_copy_instructions,
    write_instance_ids,
    write_json,
    write_predictions,
    write_run_index,
)
from .types import ScoringSmokeSubsetSummary


def prepare_scoring_smoke_subset(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    max_instances: int,
    conditions: str | list[str],
) -> ScoringSmokeSubsetSummary:
    """Create a small paired subset run from an existing full run.

    Selects the first *max_instances* instances from the source run
    and copies their predictions into a new run directory suitable
    for a quick Docker scoring smoke test.

    Args:
        source_run_dir: Full run directory to draw instances from.
        output_run_dir: Destination for the subset run.
        max_instances: Maximum number of instances to include.
        conditions: Condition specifier (must be paired baseline+llmdebug).

    Returns:
        Summary of the prepared smoke subset.

    Raises:
        ValueError: If *max_instances* < 1.
        RuntimeError: If source run is invalid or output exists.
    """
    if max_instances < 1:
        raise ValueError("max_instances must be >= 1")
    if not source_run_dir.exists():
        raise RuntimeError(f"source run directory does not exist: {source_run_dir}")

    condition_list = parse_paired_conditions(conditions)
    validate_paired_predictions(
        source_run_dir,
        conditions=condition_list,
        require_files=True,
        require_nonempty=True,
    )

    source_records = {
        condition: read_predictions(prediction_path(source_run_dir, condition))
        for condition in condition_list
    }
    source_ids = _source_instance_order(
        source_run_dir=source_run_dir, fallback_condition=condition_list[0]
    )
    if not source_ids:
        raise RuntimeError(f"source run has no instance ids: {source_run_dir}")

    selected_ids = source_ids[:max_instances]
    if output_run_dir.exists():
        existing = list(output_run_dir.iterdir())
        if existing:
            raise RuntimeError(
                f"output run directory already exists and is not empty: {output_run_dir}"
            )
    ensure_run_layout(output_run_dir, conditions=condition_list)

    for condition in condition_list:
        by_id = {record.instance_id: record for record in source_records[condition]}
        subset = [by_id[item] for item in selected_ids if item in by_id]
        if len(subset) != len(selected_ids):
            raise RuntimeError(
                f"condition {condition} does not contain all selected subset ids; "
                "cannot build scoring smoke subset"
            )
        write_predictions(prediction_path(output_run_dir, condition), subset)

    write_instance_ids(instance_ids_path(output_run_dir), selected_ids)
    write_json(
        metadata_path(output_run_dir),
        {
            "schema_version": "swebench_pro_score_smoke_subset_v1",
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_run_dir": str(source_run_dir),
            "selected_instance_count": len(selected_ids),
            "conditions": condition_list,
        },
    )
    write_run_index(
        output_run_dir,
        run_id=output_run_dir.name,
        conditions=condition_list,
    )
    write_copy_instructions(output_run_dir, conditions=condition_list)

    return ScoringSmokeSubsetSummary(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
        instance_count=len(selected_ids),
        conditions=tuple(condition_list),
    )


def _source_instance_order(*, source_run_dir: Path, fallback_condition: str) -> list[str]:
    ids_path = instance_ids_path(source_run_dir)
    if ids_path.exists():
        loaded = load_instance_ids(ids_path)
        if loaded:
            return loaded
    return [
        record.instance_id
        for record in read_predictions(prediction_path(source_run_dir, fallback_condition))
    ]
