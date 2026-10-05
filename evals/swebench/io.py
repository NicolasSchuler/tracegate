"""Filesystem and validation helpers for SWE-Bench Lite runs.

The helpers are shared with the other SWE-Bench flavour in ``evals.swebench_common.io``
(X-04); this module binds the layout and keeps the established public names.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..swebench_common import io as _common
from ..swebench_common.io import (  # noqa: F401 (re-exported)
    _RUN_ID_RE,
    SweBenchLayout,
    _assert_condition,
    collect_prediction_ids,
    copy_instructions_path,
    ensure_run_layout,
    instance_ids_path,
    load_instance_ids,
    load_json,
    metadata_path,
    parse_conditions,
    prediction_path,
    read_predictions,
    repo_root,
    run_index_path,
    validate_paired_predictions,
    validate_prediction_file,
    validate_run_id,
    write_instance_ids,
    write_json,
    write_predictions,
)

LAYOUT = SweBenchLayout(
    label="SWE-Bench Lite",
    runs_dirname="swebench_runs",
    eval_dirname="official_eval",
    scoring_script="./scripts/score_swebench_lite_official.sh",
    index_schema="swebench_handoff_index_v1",
    scoring_heading="On the Docker scoring machine:",
)


def swebench_runs_root(*, root: Path | None = None) -> Path:
    """Return ``<root>/evals/{LAYOUT.runs_dirname}`` (*root* defaults to the checkout)."""
    return _common.runs_root(LAYOUT, root=root)


def run_dir_for_id(run_id: str, *, root: Path | None = None) -> Path:
    """Return the directory of run *run_id* after validating the id."""
    return _common.run_dir_for_id(LAYOUT, run_id, root=root)


def parse_paired_conditions(raw: str | Iterable[str] | None) -> list[str]:
    """Require exactly the baseline and llmdebug conditions."""
    return _common.parse_paired_conditions(LAYOUT, raw)


def copy_instructions_text(run_dir: Path, *, conditions: Iterable[str]) -> str:
    """The handoff instructions for the Docker scoring machine."""
    return _common.copy_instructions_text(LAYOUT, run_dir, conditions=conditions)


def write_copy_instructions(run_dir: Path, *, conditions: Iterable[str]) -> Path:
    """Write the handoff instructions into *run_dir*."""
    return _common.write_copy_instructions(LAYOUT, run_dir, conditions=conditions)


def write_run_index(run_dir: Path, *, run_id: str, conditions: Iterable[str]) -> Path:
    """Write the run's ``run_index.json`` manifest."""
    return _common.write_run_index(LAYOUT, run_dir, run_id=run_id, conditions=conditions)
