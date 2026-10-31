"""Coverage-guided snapshot enrichment — extract coverage data for crash frames.

When pytest-cov is active, this module extracts coverage data (executed/missing
lines, branch stats) for files in the stack trace. Data is placed under
``pytest.coverage`` in the snapshot.

Gracefully degrades when pytest-cov is absent, ``--cov`` not passed, or
coverage data is unavailable.
"""

from __future__ import annotations

import os
from typing import Any, cast


def _match_file(measured_files: set[str], target: str) -> str | None:
    """Find the measured-file key that corresponds to *target*.

    Tries an exact match, then *target*'s realpath, then the realpath of each
    measured file (for symlinked sources).

    Returns:
        The matching key from *measured_files*, or None if nothing matches.
    """
    if target in measured_files:
        return target
    try:
        real_target = os.path.realpath(target)
    except (OSError, ValueError):
        return None
    if real_target in measured_files:
        return real_target
    # Try realpath of each measured file (for symlinked sources)
    for mf in measured_files:
        try:
            if os.path.realpath(mf) == real_target:
                return mf
        except (OSError, ValueError):
            continue
    return None


def _extract_file_coverage(
    cov: Any,
    measured_key: str,
    *,
    max_lines: int = 200,
) -> dict[str, Any] | None:
    """Extract coverage data for a single file.

    Uses ``cov.analysis2()`` for a clean executed/missing split and falls back to
    ``CoverageData.lines()`` (executed lines only, so the line rate is unknown) if
    that fails. Partially taken branches are added when branch data is available.

    Args:
        cov: The ``coverage.Coverage`` instance.
        measured_key: The file path as known to coverage.
        max_lines: If the file has more executable lines than this, return a
            summary only (counts and rate, no line lists).

    Returns:
        The file's coverage dict, or None if no data could be read.
    """
    missing_known = True
    try:
        # analysis2 returns (filename, executed, excluded, missing, formatted_missing)
        _, executed, _, missing, _ = cov.analysis2(measured_key)
    except Exception:
        # Fallback: try raw CoverageData
        try:
            cov_data = cov.get_data()
            raw_lines = cov_data.lines(measured_key)
            if raw_lines is None:
                return None
            executed = sorted(raw_lines)
            missing = []  # Placeholder to keep local variable type stable.
            missing_known = False
        except Exception:
            return None

    executed = sorted(cast(list[int], executed))
    if missing_known:
        missing = sorted(cast(list[int], missing))
        total = len(executed) + len(missing)
        line_rate = round(len(executed) / total, 4) if total > 0 else 1.0
    else:
        total = len(executed)
        line_rate = None

    result: dict[str, Any] = {}

    if total > max_lines:
        # Summary only for large files
        result["executable_count"] = total
        result["executed_count"] = len(executed)
        if missing_known:
            result["missing_count"] = len(missing)
        if line_rate is not None:
            result["line_rate"] = line_rate
    else:
        result["executed_lines"] = executed
        if missing_known:
            result["missing_lines"] = missing
        if line_rate is not None:
            result["line_rate"] = line_rate

    if not missing_known:
        result["line_rate_unknown"] = True

    # Branch stats (coverage >= 7.7)
    try:
        branch_stats = cov.branch_stats()
        if branch_stats and measured_key in branch_stats:
            file_branches = branch_stats[measured_key]
            # Only include if there are partial branches
            partial = {
                str(line): [total_exits, taken_exits]
                for line, (total_exits, taken_exits) in file_branches.items()
                if taken_exits < total_exits
            }
            if partial:
                result["partial_branches"] = partial
    except (AttributeError, TypeError):
        pass  # Old coverage version or no branch data

    return result


def get_coverage_for_frames(
    cov: Any,
    frame_files: list[str],
    *,
    max_lines: int = 200,
) -> dict[str, dict[str, Any]] | None:
    """Extract coverage data for the files that appear in a stack trace.

    Args:
        cov: The ``coverage.Coverage`` instance from pytest-cov.
        frame_files: File paths from the traceback; duplicates are ignored.
        max_lines: Token budget; files with more executable lines get a summary
            without line lists.

    Returns:
        A dict mapping each matched frame path (as given) to its coverage data, or
        None if the coverage data is unreadable or empty, or no frame matched.
    """
    try:
        cov_data = cov.get_data()
        measured_files = set(cov_data.measured_files())
    except Exception:
        return None

    if not measured_files:
        return None

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_files: list[str] = []
    for f in frame_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    result: dict[str, dict[str, Any]] = {}
    for target in unique_files:
        matched = _match_file(measured_files, target)
        if matched is None:
            continue
        file_cov = _extract_file_coverage(cov, matched, max_lines=max_lines)
        if file_cov is not None:
            result[target] = file_cov

    return result if result else None


def extract_coverage_from_plugin(
    item: Any,
    tb: Any,
    warnings: list[str],
) -> dict[str, dict[str, Any]] | None:
    """Collect pytest-cov coverage data for the files in a failing test's traceback.

    This is the entry point used by the pytest plugin. It returns None without
    touching coverage when ``LLMDEBUG_INCLUDE_COVERAGE`` is set to anything other
    than ``1``/``true``/``yes``/``on`` or pytest-cov is not active, and saves the
    in-memory coverage data before reading it.

    Args:
        item: The failing test item; its config locates the pytest-cov plugin.
        tb: The failure's traceback; its frames select the files to report.
        warnings: List that collects warning messages for failed extraction steps.

    Returns:
        A dict mapping traceback file paths to coverage data, or None when coverage
        is disabled, unavailable, or has no data for those files.
    """
    import traceback as tb_module

    # Check env var opt-out
    include_cov = os.getenv("LLMDEBUG_INCLUDE_COVERAGE", "true").lower()
    if include_cov not in ("1", "true", "yes", "on"):
        return None

    # Check for pytest-cov plugin
    try:
        cov_plugin = item.config.pluginmanager.getplugin("_cov")
    except Exception:
        return None

    if cov_plugin is None:
        return None

    # Get the coverage controller
    try:
        cov_controller = cov_plugin.cov_controller
        if cov_controller is None:
            return None
        cov = cov_controller.cov
        if cov is None:
            return None
    except Exception as e:  # third-party plugin internals
        warnings.append(f"coverage extraction failed: {type(e).__name__}: {e}")
        return None

    # Flush in-memory data (idempotent, exception-safe)
    try:
        cov.save()
    except Exception:  # nosec B110 — best-effort save
        pass  # analysis2() may still work with stale data

    # Extract frame files from traceback
    try:
        extracted = tb_module.extract_tb(tb)
        frame_files = [frame.filename for frame in extracted if frame.filename]
    except Exception as e:
        warnings.append(f"coverage frame extraction failed: {type(e).__name__}: {e}")
        return None

    if not frame_files:
        return None

    try:
        return get_coverage_for_frames(cov, frame_files)
    except Exception as e:
        warnings.append(f"coverage data extraction failed: {type(e).__name__}: {e}")
        return None
