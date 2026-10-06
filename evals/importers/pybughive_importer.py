"""Strict reproducibility importer for PyBugHive.

Uses offline dumps as the authoritative source. No git/network fallback is used.
"""

from __future__ import annotations

import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency
from evals.importers.base import count_patch_lines, generate_oracle_patch, validate_generated_case
from evals.importers.native_repro_helpers import (
    SKIP_UNSAFE_DATASET_PATH,
    SKIP_UNSUPPORTED_NATIVE_COMMAND,
    SKIP_UNSUPPORTED_PATCH_FILE_COUNT,
    normalize_native_pytest_command,
    normalize_relative_path,
    write_native_case_files,
)
from evals.importers.types import ImportRunSummary

from .. import _json as json


@dataclass(frozen=True)
class PyBugHiveIssueSpec:
    username: str
    repository: str
    issue_id: int
    title: str
    fix_commit: str
    parent_commit: str
    changed_python_files: list[str]
    test_steps: str
    test_steps_full: str

    @property
    def project_key(self) -> str:
        return f"{self.username}/{self.repository}"


def _normalize_parent_commit(value: str) -> str:
    token = value.strip().split()[0] if value.strip() else ""
    return token


def _looks_like_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    return "/tests/" in f"/{normalized}" or name.startswith("test_")


def _load_issue_specs(pybughive_repo: Path) -> list[PyBugHiveIssueSpec]:
    dataset_path = pybughive_repo / "dataset" / "pybughive_current.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    out: list[PyBugHiveIssueSpec] = []

    if not isinstance(payload, list):
        return out

    for project in payload:
        if not isinstance(project, dict):
            continue
        username = str(project.get("username", "")).strip()
        repository = str(project.get("repository", "")).strip()
        issues = project.get("issues")
        if not username or not repository or not isinstance(issues, list):
            continue

        for issue in issues:
            if not isinstance(issue, dict):
                continue
            issue_id = issue.get("id")
            title = str(issue.get("title", "")).strip()
            commits = issue.get("commits")
            if not isinstance(issue_id, int) or not isinstance(commits, list) or not commits:
                continue

            commit = commits[0]
            if not isinstance(commit, dict):
                continue

            fix_commit = str(commit.get("hash", "")).strip()
            parent_commit = _normalize_parent_commit(str(commit.get("parents", "")))
            if not fix_commit or not parent_commit:
                continue

            stat = commit.get("stat")
            files = stat.get("files", []) if isinstance(stat, dict) else []
            changed_python_files: list[str] = []
            for file_item in files:
                if not isinstance(file_item, dict):
                    continue
                filename = str(file_item.get("filename", "")).strip()
                normalized = normalize_relative_path(filename)
                if normalized and normalized.endswith(".py"):
                    changed_python_files.append(normalized)

            out.append(
                PyBugHiveIssueSpec(
                    username=username,
                    repository=repository,
                    issue_id=issue_id,
                    title=title,
                    fix_commit=fix_commit,
                    parent_commit=parent_commit,
                    changed_python_files=changed_python_files,
                    test_steps=str(issue.get("testSteps", "") or "").strip(),
                    test_steps_full=str(issue.get("testStepsFull", "") or "").strip(),
                )
            )

    return out


def _choose_target_source_file(spec: PyBugHiveIssueSpec) -> str | None:
    non_test = [path for path in spec.changed_python_files if not _looks_like_test_path(path)]
    if len(non_test) == 1:
        return non_test[0]
    if len(spec.changed_python_files) == 1:
        return spec.changed_python_files[0]
    return None


def _build_archive_member_index(
    offline_archives: dict[str, Path],
) -> tuple[dict[Path, tuple[str, ...]], int]:
    index: dict[Path, tuple[str, ...]] = {}
    scan_failures = 0
    unique_paths = sorted(set(offline_archives.values()), key=lambda path: str(path))
    for archive_path in unique_paths:
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                members = tuple(name for name in archive.namelist() if not name.endswith("/"))
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
            scan_failures += 1
            continue
        index[archive_path] = members
    return index, scan_failures


def _resolve_archive_member(
    archive_member_index: dict[Path, tuple[str, ...]],
    *,
    commitish: str,
    rel_path: str,
) -> tuple[tuple[Path, str] | None, str | None]:
    normalized = normalize_relative_path(rel_path)
    if normalized is None:
        return None, SKIP_UNSAFE_DATASET_PATH

    suffix = f"/{normalized}"
    all_matches: list[tuple[Path, str]] = []
    for archive_path, members in archive_member_index.items():
        all_matches.extend(
            (archive_path, member)
            for member in members
            if member == normalized or member.endswith(suffix)
        )
    if not all_matches:
        return None, "offline_dump_path_not_found"

    commit_matches = [entry for entry in all_matches if commitish in entry[1]]
    if len(commit_matches) == 1:
        return commit_matches[0], None
    if len(commit_matches) > 1:
        return None, "offline_dump_path_ambiguous"
    if len(all_matches) == 1:
        return all_matches[0], None
    return None, "offline_dump_path_ambiguous"


def _read_archive_file(
    archive_member_index: dict[Path, tuple[str, ...]],
    *,
    commitish: str,
    rel_path: str,
) -> tuple[str | None, str | None]:
    chosen, reason = _resolve_archive_member(
        archive_member_index,
        commitish=commitish,
        rel_path=rel_path,
    )
    if reason is not None:
        return None, reason
    if chosen is None:
        return None, "offline_dump_path_not_found"

    archive_path, member = chosen
    try:
        with zipfile.ZipFile(archive_path, "r") as archive, archive.open(member, "r") as handle:
            data = handle.read().decode("utf-8", errors="replace")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, KeyError):
        return None, "offline_dump_archive_scan_failed"
    else:
        return data, None


def _build_case_metadata(
    *,
    spec: PyBugHiveIssueSpec,
    native_command,
    fix_lines: int,
) -> dict[str, Any]:
    difficulty = "easy" if fix_lines <= 3 else ("medium" if fix_lines <= 10 else "hard")
    category = "logic_error"
    return {
        "description": f"PyBugHive {spec.project_key} issue {spec.issue_id}: {spec.title}",
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": infer_snapshot_dependency(category),
        "expected_exception": "AssertionError",
        "tags": ["pybughive", spec.repository],
        "python_args": list(native_command.python_args),
        "timeout_sec": 30,
        "bug_source": "pybughive",
        "contamination_risk": "medium",
        "expected_fix_lines": fix_lines,
        "source_id": f"pybughive_{spec.repository}_{spec.issue_id}",
        "difficulty_rationale": "Estimated from patch line count.",
        "repro_mode": "native_dataset_test",
        "source_test_command": native_command.source_test_command,
        "source_test_selector": native_command.source_test_selector,
        "source_buggy_commit": spec.parent_commit,
        "source_fixed_commit": spec.fix_commit,
        "pybughive_project": spec.project_key,
        "pybughive_issue_id": spec.issue_id,
    }


def import_pybughive(
    output_dir: Path,
    *,
    pybughive_repo: Path,
    offline_archives: dict[str, Path],
    data_root: Path,
    max_cases: int = 0,
    validate: bool = False,
    dry_run: bool = False,
    jobs: int = 1,
) -> ImportRunSummary:
    """Import PyBugHive with offline dumps as authoritative artifact source."""
    del data_root  # Reserved for future tooling assets.
    del jobs

    specs = _load_issue_specs(pybughive_repo)
    skip_reasons: Counter[str] = Counter()
    imported_case_ids: list[str] = []
    attempted = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    if not offline_archives:
        attempted_total = len(specs)
        missing_reasons = {"offline_dumps_missing": attempted_total} if attempted_total > 0 else {}
        return ImportRunSummary(
            source="pybughive",
            attempted=attempted_total,
            imported=0,
            skipped=attempted_total,
            skip_reasons=missing_reasons,
            imported_case_ids=[],
            notes=["Strict reproducibility mode: offline dumps are mandatory and authoritative."],
        )

    archive_member_index, scan_failures = _build_archive_member_index(offline_archives)
    if not archive_member_index:
        attempted_total = len(specs)
        missing_reasons = (
            {"offline_dump_archive_scan_failed": attempted_total} if attempted_total > 0 else {}
        )
        return ImportRunSummary(
            source="pybughive",
            attempted=attempted_total,
            imported=0,
            skipped=attempted_total,
            skip_reasons=missing_reasons,
            imported_case_ids=[],
            notes=["Strict reproducibility mode: offline dumps are mandatory and authoritative."],
        )

    for spec in specs:
        if max_cases > 0 and len(imported_case_ids) >= max_cases:
            break

        attempted += 1
        case_id = f"pbh_{spec.repository}_{spec.issue_id}".replace("-", "_").lower()
        case_dir = output_dir / case_id
        if case_dir.exists() and not dry_run:
            skip_reasons["case_already_exists"] += 1
            continue

        command_text = spec.test_steps or spec.test_steps_full
        native_command = normalize_native_pytest_command(command_text)
        if native_command is None:
            skip_reasons[SKIP_UNSUPPORTED_NATIVE_COMMAND] += 1
            continue

        target_rel_file = _choose_target_source_file(spec)
        if target_rel_file is None:
            skip_reasons[SKIP_UNSUPPORTED_PATCH_FILE_COUNT] += 1
            continue

        buggy_code, buggy_reason = _read_archive_file(
            archive_member_index,
            commitish=spec.parent_commit,
            rel_path=target_rel_file,
        )
        fixed_code, fixed_reason = _read_archive_file(
            archive_member_index,
            commitish=spec.fix_commit,
            rel_path=target_rel_file,
        )
        if buggy_code is None or fixed_code is None:
            skip_reasons[buggy_reason or fixed_reason or "offline_dump_path_not_found"] += 1
            continue

        test_files: dict[str, str] = {}
        missing_test = False
        for test_path in native_command.test_paths:
            test_content, test_reason = _read_archive_file(
                archive_member_index,
                commitish=spec.fix_commit,
                rel_path=test_path,
            )
            if test_content is None:
                test_content, test_reason = _read_archive_file(
                    archive_member_index,
                    commitish=spec.parent_commit,
                    rel_path=test_path,
                )
            if test_content is None:
                skip_reasons[test_reason or "native_test_file_not_found"] += 1
                missing_test = True
                break
            normalized_test_path = normalize_relative_path(test_path)
            if normalized_test_path is None:
                skip_reasons[SKIP_UNSAFE_DATASET_PATH] += 1
                missing_test = True
                break
            test_files[normalized_test_path] = test_content
        if missing_test:
            continue

        oracle_patch = generate_oracle_patch(buggy_code, fixed_code, filename=target_rel_file)
        if not oracle_patch.strip():
            skip_reasons["empty_oracle_patch"] += 1
            continue

        fix_lines = count_patch_lines(oracle_patch)
        metadata = _build_case_metadata(
            spec=spec, native_command=native_command, fix_lines=fix_lines
        )

        if dry_run:
            imported_case_ids.append(case_id)
            continue

        try:
            write_native_case_files(
                case_dir,
                buggy_files={target_rel_file: buggy_code},
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
        "Offline dumps are authoritative; no git fallback is used.",
        "No synthetic marker-test conversion is used.",
    ]
    if scan_failures > 0:
        notes.append(f"Ignored {scan_failures} unreadable offline archive(s) during index scan.")
    return ImportRunSummary(
        source="pybughive",
        attempted=attempted,
        imported=imported,
        skipped=skipped,
        skip_reasons=dict(skip_reasons),
        imported_case_ids=imported_case_ids,
        notes=notes,
    )
