"""Strict reproducibility importer for TrickyBugs."""

from __future__ import annotations

import re
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency
from evals.importers.base import count_patch_lines, generate_oracle_patch, validate_generated_case
from evals.importers.native_repro_helpers import (
    SKIP_UNSAFE_DATASET_PATH,
    SKIP_UNSUPPORTED_NATIVE_COMMAND,
    normalize_native_pytest_command,
    normalize_relative_path,
    write_native_case_files,
)
from evals.importers.types import ImportRunSummary

from .. import _json as json

_JSON_SCAN_BYTES_LIMIT = 10_000_000
_PYTHON_LANGS = {"python", "python3", "py"}
_LANG_KEYS = ("language", "lang", "programming_language")
_BUGGY_KEYS = (
    "buggy_file",
    "buggy_path",
    "bug_path",
    "faulty_file",
    "faulty_path",
    "before_file",
    "before_path",
)
_FIXED_KEYS = (
    "fixed_file",
    "fixed_path",
    "fix_file",
    "fix_path",
    "correct_file",
    "correct_path",
    "after_file",
    "after_path",
)
_TEST_COMMAND_KEYS = (
    "test_command",
    "test_cmd",
    "run_test",
    "run_tests",
    "testSteps",
    "test_steps",
    "test",
)
_PID_KEYS = ("pid", "problem_id", "problemId")
_PROGRAM_KEYS = ("program_id", "programId", "id", "uid")


@dataclass(frozen=True)
class TrickyBugsCandidate:
    pid: str
    program_id: str
    buggy_rel_path: str
    fixed_rel_path: str
    test_command: str | None
    origin: str


def _safe_slug(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip()).strip("_").lower()
    return slug or "unknown"


def _first_string(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _iter_dict_nodes(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_dict_nodes(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_dict_nodes(item)


def _candidate_from_record(
    *,
    record: dict[str, Any],
    origin: str,
) -> TrickyBugsCandidate | None:
    buggy_raw = _first_string(record, _BUGGY_KEYS)
    fixed_raw = _first_string(record, _FIXED_KEYS)
    if buggy_raw is None or fixed_raw is None:
        return None

    language = (_first_string(record, _LANG_KEYS) or "").strip().lower()
    if language and language not in _PYTHON_LANGS:
        return None

    buggy_rel = normalize_relative_path(buggy_raw)
    fixed_rel = normalize_relative_path(fixed_raw)
    if buggy_rel is None or fixed_rel is None:
        return None
    if not buggy_rel.endswith(".py") or not fixed_rel.endswith(".py"):
        return None

    command = _first_string(record, _TEST_COMMAND_KEYS)

    pid = _safe_slug(_first_string(record, _PID_KEYS) or "unknown")
    program_id = _safe_slug(_first_string(record, _PROGRAM_KEYS) or "unknown")
    return TrickyBugsCandidate(
        pid=pid,
        program_id=program_id,
        buggy_rel_path=buggy_rel,
        fixed_rel_path=fixed_rel,
        test_command=command,
        origin=origin,
    )


def _iter_python_candidates(trickybugs_root: Path) -> list[TrickyBugsCandidate]:
    discovered: list[TrickyBugsCandidate] = []
    for json_path in sorted(trickybugs_root.rglob("*.json")):
        try:
            if json_path.stat().st_size > _JSON_SCAN_BYTES_LIMIT:
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        except json.JSONDecodeError:
            continue

        origin = f"json:{json_path.relative_to(trickybugs_root)}"
        for record in _iter_dict_nodes(payload):
            candidate = _candidate_from_record(record=record, origin=origin)
            if candidate is not None:
                discovered.append(candidate)

    deduped: dict[tuple[str, str, str, str, str], TrickyBugsCandidate] = {}
    for candidate in discovered:
        key = (
            candidate.pid,
            candidate.program_id,
            candidate.buggy_rel_path,
            candidate.fixed_rel_path,
            candidate.test_command or "",
        )
        deduped.setdefault(key, candidate)

    return sorted(
        deduped.values(),
        key=lambda c: (
            c.pid,
            c.program_id,
            c.buggy_rel_path,
            c.fixed_rel_path,
            c.test_command or "",
        ),
    )


def _read_relative_file(root: Path, rel_path: str) -> str | None:
    normalized = normalize_relative_path(rel_path)
    if normalized is None:
        return None
    file_path = root / normalized
    if not file_path.exists() or not file_path.is_file():
        return None
    return file_path.read_text(encoding="utf-8", errors="replace")


def import_trickybugs(
    output_dir: Path,
    *,
    trickybugs_root: Path,
    data_root: Path,
    max_cases: int = 0,
    validate: bool = False,
    dry_run: bool = False,
    jobs: int = 1,
) -> ImportRunSummary:
    """Import TrickyBugs using strict native dataset test command mapping."""
    del data_root
    del jobs

    candidates = _iter_python_candidates(trickybugs_root)
    skip_reasons: Counter[str] = Counter()
    imported_case_ids: list[str] = []
    attempted = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        if max_cases > 0 and len(imported_case_ids) >= max_cases:
            break

        attempted += 1
        case_id = f"tb_{_safe_slug(candidate.pid)}_{_safe_slug(candidate.program_id)}"
        case_dir = output_dir / case_id
        if case_dir.exists() and not dry_run:
            skip_reasons["case_already_exists"] += 1
            continue

        if not candidate.test_command:
            skip_reasons["missing_native_test_oracle_mapping"] += 1
            continue

        native_command = normalize_native_pytest_command(candidate.test_command)
        if native_command is None:
            skip_reasons[SKIP_UNSUPPORTED_NATIVE_COMMAND] += 1
            continue

        buggy_code = _read_relative_file(trickybugs_root, candidate.buggy_rel_path)
        fixed_code = _read_relative_file(trickybugs_root, candidate.fixed_rel_path)
        if buggy_code is None:
            skip_reasons["missing_buggy_source"] += 1
            continue
        if fixed_code is None:
            skip_reasons["missing_fixed_source"] += 1
            continue
        if not buggy_code.strip() or not fixed_code.strip():
            skip_reasons["empty_source_file"] += 1
            continue

        test_files: dict[str, str] = {}
        missing_test = False
        for test_path in native_command.test_paths:
            normalized_test = normalize_relative_path(test_path)
            if normalized_test is None:
                skip_reasons[SKIP_UNSAFE_DATASET_PATH] += 1
                missing_test = True
                break
            test_content = _read_relative_file(trickybugs_root, normalized_test)
            if test_content is None:
                skip_reasons["native_test_file_not_found"] += 1
                missing_test = True
                break
            test_files[normalized_test] = test_content
        if missing_test:
            continue

        oracle_patch = generate_oracle_patch(
            buggy_code,
            fixed_code,
            filename=candidate.buggy_rel_path,
        )
        if not oracle_patch.strip():
            skip_reasons["empty_oracle_patch"] += 1
            continue

        fix_lines = count_patch_lines(oracle_patch)
        difficulty = "easy" if fix_lines <= 3 else ("medium" if fix_lines <= 10 else "hard")
        category = "logic_error"
        metadata: dict[str, Any] = {
            "description": (
                f"TrickyBugs PID {candidate.pid} program {candidate.program_id} "
                f"({candidate.origin})"
            ),
            "category": category,
            "difficulty": difficulty,
            "snapshot_dependency": infer_snapshot_dependency(category),
            "expected_exception": "AssertionError",
            "tags": ["trickybugs", candidate.pid],
            "python_args": list(native_command.python_args),
            "timeout_sec": 30,
            "bug_source": "trickybugs",
            "contamination_risk": "medium",
            "expected_fix_lines": fix_lines,
            "source_id": f"trickybugs_{_safe_slug(candidate.pid)}_{_safe_slug(candidate.program_id)}",
            "difficulty_rationale": "Estimated from patch line count.",
            "repro_mode": "native_dataset_test",
            "source_test_command": native_command.source_test_command,
            "source_test_selector": native_command.source_test_selector,
            "source_buggy_commit": f"path:{candidate.buggy_rel_path}",
            "source_fixed_commit": f"path:{candidate.fixed_rel_path}",
            "trickybugs_pid": candidate.pid,
            "trickybugs_program_id": candidate.program_id,
        }

        if dry_run:
            imported_case_ids.append(case_id)
            continue

        try:
            write_native_case_files(
                case_dir,
                buggy_files={candidate.buggy_rel_path: buggy_code},
                test_files=test_files,
                oracle_patch=oracle_patch,
                metadata=metadata,
            )
        except ValueError:
            skip_reasons[SKIP_UNSAFE_DATASET_PATH] += 1
            shutil.rmtree(case_dir, ignore_errors=True)
            continue

        # DAT-05: the schema is always checked; --validate also runs the case
        # (the baseline must fail and the oracle must fix it).
        ok_case, _ = validate_generated_case(case_dir, schema_only=not validate)
        if not ok_case:
            skip_reasons["schema_validation_failed"] += 1
            shutil.rmtree(case_dir, ignore_errors=True)
            continue

        imported_case_ids.append(case_id)

    imported = len(imported_case_ids)
    skipped = max(0, attempted - imported)
    notes = [
        "Strict reproducibility mode: native dataset test commands only.",
        "No synthetic marker-test conversion is used.",
        "Candidates without deterministic native test mappings are skipped.",
    ]
    return ImportRunSummary(
        source="trickybugs",
        attempted=attempted,
        imported=imported,
        skipped=skipped,
        skip_reasons=dict(skip_reasons),
        imported_case_ids=imported_case_ids,
        notes=notes,
    )
