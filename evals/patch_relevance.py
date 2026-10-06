"""Patch-verifier relevance scoring: which files a patch is expected to touch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from evals.eval_evidence import (
    _crash_file_from_snapshot,
    _detect_failing_test_file,
    _extract_subprocess_script_neighbors,
    _import_neighbors,
)
from evals.eval_patches import (
    _extract_candidate_patch_files,
    _is_test_file_path,
    _patch_contains_substantive_change,
    _patch_mentions_crash_file,
    _patch_mentions_failing_test_context,
    _patch_mentions_scope_files,
)
from evals.eval_types import (
    PATCH_VERIFIER_MIN_SCORE,
    PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
)


def _effective_patch_verifier_scope(*, configured_scope: str, is_adaptive_condition: bool) -> str:
    """Resolve configured patch-verifier scope into an attempt-effective scope."""
    if configured_scope == PATCH_VERIFIER_SCOPE_DISABLED:
        return PATCH_VERIFIER_SCOPE_DISABLED
    if configured_scope == PATCH_VERIFIER_SCOPE_ALL:
        return PATCH_VERIFIER_SCOPE_ALL
    if configured_scope == PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY and is_adaptive_condition:
        return PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY
    return PATCH_VERIFIER_SCOPE_DISABLED


def _infer_test_source_scope(
    *,
    cwd: Path,
    crash_file: str | None,
    failing_test_file: str | None,
    enabled: bool,
    file_index_lines: Sequence[str] | None = None,
) -> list[str]:
    if not enabled:
        return []

    candidates: list[str] = []
    if isinstance(crash_file, str) and crash_file and _is_test_file_path(crash_file):
        candidates.append(crash_file)
    if (
        isinstance(failing_test_file, str)
        and failing_test_file
        and _is_test_file_path(failing_test_file)
        and failing_test_file not in candidates
    ):
        candidates.append(failing_test_file)

    inferred: list[str] = []
    for rel in candidates:
        for neighbor in _import_neighbors(root=cwd, rel_path=rel, limit=12):
            normalized = str(PurePosixPath(neighbor.replace("\\", "/")))
            if normalized not in inferred:
                inferred.append(normalized)
        for neighbor in _extract_subprocess_script_neighbors(
            root=cwd,
            rel_path=rel,
            limit=12,
            file_index_lines=file_index_lines,
        ):
            normalized = str(PurePosixPath(neighbor.replace("\\", "/")))
            if normalized not in inferred:
                inferred.append(normalized)
    return inferred


def _score_patch_relevance(
    *,
    diff: str,
    cwd: Path,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    min_score: int = PATCH_VERIFIER_MIN_SCORE,
    infer_test_source_scope: bool = True,
    file_index_lines: Sequence[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Heuristic relevance score for a proposed diff.

    Checks whether the patch touches the crash file, its import
    neighbors, the inferred source scope, or test files.  Returns
    (passed, details_dict) with ``passed=True`` when the score
    meets ``min_score``.
    """
    patch_files = _extract_candidate_patch_files(diff)
    crash_file = _crash_file_from_snapshot(snapshot)
    crash_neighbors = (
        _import_neighbors(root=cwd, rel_path=crash_file, limit=12) if crash_file else []
    )
    crash_neighbor_set = {str(PurePosixPath(path.replace("\\", "/"))) for path in crash_neighbors}
    failing_test_file = _detect_failing_test_file(
        stdout=stdout,
        stderr=stderr,
        root=cwd,
        file_index_lines=file_index_lines,
    )
    inferred_source_scope = _infer_test_source_scope(
        cwd=cwd,
        crash_file=crash_file,
        failing_test_file=failing_test_file,
        enabled=infer_test_source_scope,
        file_index_lines=file_index_lines,
    )
    inferred_source_scope_set = {
        str(PurePosixPath(path.replace("\\", "/"))) for path in inferred_source_scope
    }

    substantive_change = _patch_contains_substantive_change(diff)
    matches_crash_file = _patch_mentions_crash_file(patch_files=patch_files, crash_file=crash_file)
    matches_crash_neighbor = any(path in crash_neighbor_set for path in patch_files)
    matches_inferred_source_scope = _patch_mentions_scope_files(
        patch_files=patch_files,
        scope_files=inferred_source_scope_set,
    )
    touches_test_file = any(_is_test_file_path(path) for path in patch_files)
    touches_failing_test_file = _patch_mentions_failing_test_context(
        patch_files=patch_files,
        failing_test_file=failing_test_file,
    )

    score = 0
    reasons: list[str] = []
    if substantive_change:
        score += 1
        reasons.append("substantive_change")
    else:
        score -= 2
        reasons.append("no_substantive_change")

    if crash_file:
        if matches_crash_file:
            score += 2
            reasons.append("matches_crash_file")
        elif matches_crash_neighbor:
            score += 1
            reasons.append("matches_crash_neighbor")
        elif matches_inferred_source_scope:
            score += 1
            reasons.append("matches_inferred_source_scope")
        else:
            score -= 1
            reasons.append("misses_crash_scope")
    else:
        if matches_inferred_source_scope:
            score += 1
            reasons.append("matches_inferred_source_scope")
        else:
            reasons.append("no_crash_file_available")

    if touches_failing_test_file:
        score -= 2
        reasons.append("touches_failing_test_file")
    elif touches_test_file:
        score -= 1
        reasons.append("touches_test_file")
    else:
        score += 1
        reasons.append("non_test_edit")

    normalized_min_score = max(0, int(min_score))
    passed = bool(score >= normalized_min_score)

    return passed, {
        "patch_verifier_passed": passed,
        "patch_verifier_score": int(score),
        "patch_verifier_min_score": int(normalized_min_score),
        "patch_verifier_reasons": reasons,
        "patch_verifier_patch_files": patch_files,
        "patch_verifier_crash_file": crash_file,
        "patch_verifier_crash_file_matched": matches_crash_file,
        "patch_verifier_crash_neighbor_matched": matches_crash_neighbor,
        "patch_verifier_crash_neighbors": sorted(crash_neighbor_set),
        "patch_verifier_inferred_source_scope": sorted(inferred_source_scope_set),
        "patch_verifier_source_scope_matched": matches_inferred_source_scope,
        "patch_verifier_failing_test_file": failing_test_file,
        "patch_verifier_touches_test_file": touches_test_file,
        "patch_verifier_touches_failing_test_file": touches_failing_test_file,
        "patch_verifier_substantive_change": substantive_change,
    }
