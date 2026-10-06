"""Strict reproducibility importer for BugsInPy."""

from __future__ import annotations

import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency
from evals.importers.base import count_patch_lines, generate_oracle_patch, validate_generated_case
from evals.importers.native_repro_helpers import (
    SKIP_GIT_CLONE_FAILED,
    SKIP_GIT_SHOW_FAILED,
    SKIP_UNSAFE_DATASET_PATH,
    SKIP_UNSUPPORTED_NATIVE_COMMAND,
    SKIP_UNSUPPORTED_PATCH_FILE_COUNT,
    GitShowFileRequest,
    ensure_git_repo,
    git_show_file,
    normalize_native_pytest_command,
    normalize_relative_path,
    write_native_case_files,
)
from evals.importers.types import ImportRunSummary

_PROJECT_INFO_RE = re.compile(r'^([a-zA-Z_]+)\s*=\s*"?(.*?)"?\s*$')
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")


@dataclass(frozen=True)
class BugsInPyBugSpec:
    project_name: str
    bug_id: str
    github_url: str
    buggy_commit_id: str
    fixed_commit_id: str
    test_file: str
    run_test_cmd: str
    patch_text: str
    pythonpath: str


def _parse_key_value_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROJECT_INFO_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        out[key] = value
    return out


def _extract_python_files_from_patch(patch_text: str) -> list[str]:
    files: list[str] = []
    for line in patch_text.splitlines():
        match = _DIFF_FILE_RE.match(line.strip())
        if not match:
            continue
        a_file = match.group(1).strip()
        b_file = match.group(2).strip()
        candidate = b_file if b_file.endswith(".py") else a_file
        normalized = normalize_relative_path(candidate)
        if normalized is None or not normalized.endswith(".py"):
            continue
        if normalized not in files:
            files.append(normalized)
    return files


def _iter_bugs(bugsinpy_repo: Path) -> list[BugsInPyBugSpec]:
    projects_root = bugsinpy_repo / "projects"
    if not projects_root.exists():
        return []

    out: list[BugsInPyBugSpec] = []
    for project_dir in sorted(projects_root.iterdir()):
        if not project_dir.is_dir():
            continue

        project_info_path = project_dir / "project.info"
        if not project_info_path.exists():
            continue
        project_info = _parse_key_value_file(project_info_path)
        github_url = project_info.get("github_url", "").strip()
        if not github_url:
            continue

        bugs_dir = project_dir / "bugs"
        if not bugs_dir.exists():
            continue

        for bug_dir in sorted(p for p in bugs_dir.iterdir() if p.is_dir()):
            bug_info_path = bug_dir / "bug.info"
            run_test_path = bug_dir / "run_test.sh"
            patch_path = bug_dir / "bug_patch.txt"
            if not (bug_info_path.exists() and run_test_path.exists() and patch_path.exists()):
                continue

            bug_info = _parse_key_value_file(bug_info_path)
            buggy_commit_id = bug_info.get("buggy_commit_id", "").strip()
            fixed_commit_id = bug_info.get("fixed_commit_id", "").strip()
            if not buggy_commit_id or not fixed_commit_id:
                continue

            out.append(
                BugsInPyBugSpec(
                    project_name=project_dir.name,
                    bug_id=bug_dir.name,
                    github_url=github_url,
                    buggy_commit_id=buggy_commit_id,
                    fixed_commit_id=fixed_commit_id,
                    test_file=str(bug_info.get("test_file", "")).strip().strip('"'),
                    run_test_cmd=run_test_path.read_text(encoding="utf-8", errors="replace"),
                    patch_text=patch_path.read_text(encoding="utf-8", errors="replace"),
                    pythonpath=str(bug_info.get("pythonpath", "")).strip().strip('"'),
                )
            )
    return out


def _build_case_metadata(
    *,
    spec: BugsInPyBugSpec,
    native_command,
    fix_lines: int,
) -> dict[str, Any]:
    difficulty = "easy" if fix_lines <= 3 else ("medium" if fix_lines <= 10 else "hard")
    category = "logic_error"
    return {
        "description": f"BugsInPy {spec.project_name} bug {spec.bug_id}",
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": infer_snapshot_dependency(category),
        "expected_exception": "AssertionError",
        "tags": ["bugsinpy", spec.project_name],
        "python_args": list(native_command.python_args),
        "timeout_sec": 30,
        "bug_source": "bugsinpy",
        "contamination_risk": "medium",
        "expected_fix_lines": fix_lines,
        "source_id": f"bugsinpy_{spec.project_name}_{spec.bug_id}",
        "difficulty_rationale": "Estimated from patch line count.",
        "repro_mode": "native_dataset_test",
        "source_test_command": native_command.source_test_command,
        "source_test_selector": native_command.source_test_selector,
        "source_buggy_commit": spec.buggy_commit_id,
        "source_fixed_commit": spec.fixed_commit_id,
        "bugsinpy_project": spec.project_name,
        "bugsinpy_bug_id": spec.bug_id,
    }


def import_bugsinpy(
    output_dir: Path,
    *,
    bugsinpy_repo: Path,
    data_root: Path,
    max_cases: int = 0,
    validate: bool = False,
    dry_run: bool = False,
    jobs: int = 1,
) -> ImportRunSummary:
    """Import BugsInPy using strict native dataset test commands."""
    del jobs  # Reserved for future parallel conversion.

    specs = _iter_bugs(bugsinpy_repo)
    skip_reasons: Counter[str] = Counter()
    imported_case_ids: list[str] = []
    attempted = 0

    repos_root = data_root / "repos"
    repos_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        if max_cases > 0 and len(imported_case_ids) >= max_cases:
            break

        attempted += 1
        case_id = f"bip_{spec.project_name}_{spec.bug_id}".replace("-", "_").lower()
        case_dir = output_dir / case_id
        if case_dir.exists() and not dry_run:
            skip_reasons["case_already_exists"] += 1
            continue

        if spec.pythonpath and spec.pythonpath not in {".", ""}:
            skip_reasons["unsupported_pythonpath"] += 1
            continue

        native_command = normalize_native_pytest_command(spec.run_test_cmd)
        if native_command is None:
            skip_reasons[SKIP_UNSUPPORTED_NATIVE_COMMAND] += 1
            continue

        changed_py_files = _extract_python_files_from_patch(spec.patch_text)
        if len(changed_py_files) != 1:
            skip_reasons[SKIP_UNSUPPORTED_PATCH_FILE_COUNT] += 1
            continue
        target_rel_file = changed_py_files[0]

        repo_dir = repos_root / spec.project_name
        ok, reason = ensure_git_repo(spec.github_url, repo_dir)
        if not ok:
            skip_reasons[reason or SKIP_GIT_CLONE_FAILED] += 1
            continue

        buggy_code = git_show_file(
            repo_dir,
            GitShowFileRequest(commitish=spec.buggy_commit_id, rel_path=target_rel_file),
        )
        fixed_code = git_show_file(
            repo_dir,
            GitShowFileRequest(commitish=spec.fixed_commit_id, rel_path=target_rel_file),
        )
        if buggy_code is None or fixed_code is None:
            skip_reasons[SKIP_GIT_SHOW_FAILED] += 1
            continue

        test_files: dict[str, str] = {}
        missing_test = False
        for test_path in native_command.test_paths:
            normalized_path = normalize_relative_path(test_path)
            if normalized_path is None:
                skip_reasons[SKIP_UNSAFE_DATASET_PATH] += 1
                missing_test = True
                break
            # Reproduction usually depends on the bug-fixing tests; prefer fixed commit view.
            test_content = git_show_file(
                repo_dir,
                GitShowFileRequest(commitish=spec.fixed_commit_id, rel_path=normalized_path),
            )
            if test_content is None:
                test_content = git_show_file(
                    repo_dir,
                    GitShowFileRequest(commitish=spec.buggy_commit_id, rel_path=normalized_path),
                )
            if test_content is None:
                skip_reasons["native_test_file_not_found"] += 1
                missing_test = True
                break
            test_files[normalized_path] = test_content
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
        "No synthetic marker-test conversion is used.",
    ]
    return ImportRunSummary(
        source="bugsinpy",
        attempted=attempted,
        imported=imported,
        skipped=skipped,
        skip_reasons=dict(skip_reasons),
        imported_case_ids=imported_case_ids,
        notes=notes,
    )
