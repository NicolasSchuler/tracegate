"""Discover and validate the eval cases in a cases directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals import _json as json
from evals.eval_types import (
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    VALID_SNAPSHOT_DEPENDENCIES,
    CaseSpec,
)


def load_cases(cases_dir: Path) -> list[CaseSpec]:
    """Load and validate all eval cases from *cases_dir*.

    Each subdirectory must contain a ``case.json`` with required fields
    (category, difficulty, snapshot_dependency, expected_exception).

    Raises:
        ValueError: If any case metadata is missing or invalid.
    """
    if not cases_dir.exists():
        return []

    out: list[CaseSpec] = []
    for child in sorted(cases_dir.iterdir()):
        if not child.is_dir():
            continue
        case_id = child.name
        meta_path = child / "case.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"{case_id}: invalid JSON in case.json: {e}") from e
            except OSError as e:
                raise ValueError(f"{case_id}: cannot read case.json: {e}") from e

        category = meta.get("category")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            raise ValueError(f"{case_id}: category must be one of {sorted(VALID_CATEGORIES)}")

        difficulty = meta.get("difficulty")
        if not isinstance(difficulty, str) or difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"{case_id}: difficulty must be one of {sorted(VALID_DIFFICULTIES)}")

        snapshot_dependency = meta.get("snapshot_dependency")
        if (
            not isinstance(snapshot_dependency, str)
            or snapshot_dependency not in VALID_SNAPSHOT_DEPENDENCIES
        ):
            raise ValueError(
                f"{case_id}: snapshot_dependency must be one of {sorted(VALID_SNAPSHOT_DEPENDENCIES)}"
            )

        expected_exception = meta.get("expected_exception")
        if not isinstance(expected_exception, str) or not expected_exception:
            raise ValueError(f"{case_id}: expected_exception must be a non-empty string")

        tags_raw = meta.get("tags")
        if tags_raw is None:
            tags = []
        elif isinstance(tags_raw, list) and all(isinstance(x, str) and x for x in tags_raw):
            tags = list(tags_raw)
        else:
            raise ValueError(f"{case_id}: tags must be an array of non-empty strings")

        python_args_raw = meta.get("python_args")
        if (
            not isinstance(python_args_raw, list)
            or not python_args_raw
            or not all(isinstance(x, str) and x for x in python_args_raw)
        ):
            python_args = ["-m", "pytest", "-q"]
        else:
            python_args = list(python_args_raw)

        timeout_sec_raw = meta.get("timeout_sec")
        if not isinstance(timeout_sec_raw, int) or timeout_sec_raw <= 0:
            timeout_sec = 120
        else:
            timeout_sec = timeout_sec_raw

        description = meta.get("description") if isinstance(meta.get("description"), str) else None

        # Extended optional fields for imported cases.
        bug_source = meta.get("bug_source") if isinstance(meta.get("bug_source"), str) else None
        contamination_risk = (
            meta.get("contamination_risk")
            if isinstance(meta.get("contamination_risk"), str)
            else None
        )
        expected_fix_lines_raw = meta.get("expected_fix_lines")
        expected_fix_lines = (
            expected_fix_lines_raw
            if isinstance(expected_fix_lines_raw, int) and expected_fix_lines_raw > 0
            else None
        )
        source_id = meta.get("source_id") if isinstance(meta.get("source_id"), str) else None
        difficulty_rationale = (
            meta.get("difficulty_rationale")
            if isinstance(meta.get("difficulty_rationale"), str)
            else None
        )

        out.append(
            CaseSpec(
                case_id=case_id,
                path=child,
                category=category,
                difficulty=difficulty,
                snapshot_dependency=snapshot_dependency,
                expected_exception=expected_exception,
                tags=tags,
                python_args=python_args,
                timeout_sec=timeout_sec,
                description=description,
                bug_source=bug_source,
                contamination_risk=contamination_risk,
                expected_fix_lines=expected_fix_lines,
                source_id=source_id,
                difficulty_rationale=difficulty_rationale,
            )
        )
    return out
