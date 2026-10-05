"""Apply find/replace edits and full-file answers to the working tree, with whitespace-tolerant recovery."""

from __future__ import annotations

import difflib
import os
import re
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, cast

from evals.eval_patch_constants import FORCED_PATCH_MARKER
from evals.eval_patches import _build_diff_from_changes, _is_test_file_path
from evals.patchers.openai_diff_text import _looks_like_raw_unified_diff
from evals.patchers.openai_response import (
    _extract_first_fenced_code_block,
    _get_first_choice_content,
    _get_first_choice_finish_reason,
    _safe_trunc,
)
from evals.patchers.payload_types import PatcherMeta


def _normalize_edits_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    edits = payload.get("edits")
    if isinstance(edits, list) and edits:
        return edits, "edits_list"
    if isinstance(edits, dict) and _looks_like_edit_entry(edits):
        return [edits], "edits_object"

    if _looks_like_edit_entry(payload):
        return [payload], "single_edit_root"

    edit = payload.get("edit")
    if isinstance(edit, dict) and _looks_like_edit_entry(edit):
        return [edit], "edit_object"

    changes = payload.get("changes")
    if isinstance(changes, list) and changes:
        return changes, "changes_list"

    return None, None


def _looks_like_edit_entry(entry: dict[str, Any]) -> bool:
    has_path = "path" in entry
    has_find = "find" in entry or "old" in entry
    has_replace = "replace" in entry or "new" in entry
    return has_path and has_find and has_replace


@dataclass(frozen=True)
class _ResolvedEdits:
    """Edits applied in memory; ``failure`` names the first entry that couldn't be."""

    changed: list[tuple[str, str, str]]
    recovery_modes: list[str]
    failure: str = ""
    failure_meta: PatcherMeta = dataclass_field(default_factory=lambda: cast(PatcherMeta, {}))


def _read_edit_target(path: Path) -> tuple[str, bool]:
    """Return the file as LF text and whether it uses CRLF throughout (LLM-09)."""
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw and raw.count("\r\n") == raw.count("\n")
    return _normalize_newlines(raw), crlf


def _with_line_endings(text: str, crlf: bool) -> str:
    return text.replace("\n", "\r\n") if crlf else text


def _resolve_edits(edits: list[Any], *, cwd: Path) -> _ResolvedEdits:
    """Apply an edits payload to in-memory copies of its files.

    Shared by the sync and async edits paths. Edits to one file are chained, so each sees
    the previous ones and the file gets a single diff. Matching runs on LF text and CRLF
    files keep CRLF in the diff (LLM-09). ``find`` must be non-empty and a ``count`` below
    1 means 1 (LLM-05).
    """
    originals: dict[str, tuple[str, bool]] = {}
    current: dict[str, str] = {}
    modes: list[str] = []

    def _fail(reason: str, **extra: Any) -> _ResolvedEdits:
        return _ResolvedEdits([], modes, failure=reason, failure_meta=cast(PatcherMeta, extra))

    for entry in edits:
        if not isinstance(entry, dict):
            return _fail("bad_edit_entry", entry=_safe_trunc(entry))
        entry_map = cast(dict[str, Any], entry)
        path = entry_map.get("path")
        find = entry_map.get("find", entry_map.get("old"))
        replace = entry_map.get("replace", entry_map.get("new"))
        count_raw = entry_map.get("count", 1)
        count = max(1, count_raw) if isinstance(count_raw, int) else 1
        if not isinstance(path, str) or not path.strip():
            return _fail("bad_path", entry=_safe_trunc(entry))
        if not isinstance(find, str) or not isinstance(replace, str):
            return _fail("bad_find_replace", entry=_safe_trunc(entry))
        if not find:
            return _fail("empty_find", path=path)
        if path not in current:
            file_path = (cwd / path).resolve()
            try:
                file_path.relative_to(cwd.resolve())
            except ValueError:
                return _fail("path_outside_case", path=path)
            if not file_path.exists():
                return _fail("file_not_found", path=path)
            if not file_path.is_file():
                return _fail("path_not_file", path=path)
            try:
                text, crlf = _read_edit_target(file_path)
            except Exception as exc:
                return _fail("file_read_error", path=path, exc=f"{type(exc).__name__}: {exc}")
            originals[path] = (text, crlf)
            current[path] = text
        after, match_mode = _apply_find_replace_with_recovery(
            before=current[path], find=find, replace=replace, count=count
        )
        if after is None:
            return _fail("find_not_found", path=path, find_preview=find[:200])
        normalized_mode = match_mode.removesuffix("_noop")
        if normalized_mode != "exact":
            modes.append(normalized_mode)
        current[path] = after
    changed = [
        (path, _with_line_endings(text, crlf), _with_line_endings(current[path], crlf))
        for path, (text, crlf) in originals.items()
        if current[path] != text
    ]
    return _ResolvedEdits(changed, modes)


def _record_find_recovery(meta: PatcherMeta, modes: list[str]) -> None:
    if not modes:
        meta["find_recovery_applied"] = False
        return
    unique_modes = sorted(set(modes))
    meta["find_recovery_applied"] = True
    meta["find_recovery_mode"] = unique_modes[0] if len(unique_modes) == 1 else "mixed"
    meta["find_recovery_modes"] = unique_modes


def _apply_find_replace_with_recovery(
    *,
    before: str,
    find: str,
    replace: str,
    count: int,
) -> tuple[str | None, str]:
    # LLM-05: an empty find matches everywhere, and `count or -1` made 0 mean "all".
    if not find:
        return None, "none"
    replace_count = max(1, count)

    if find in before:
        after = before.replace(find, replace, replace_count)
        if after != before:
            return after, "exact"
        return before, "exact_noop"

    before_nl = _normalize_newlines(before)
    find_nl = _normalize_newlines(find)
    replace_nl = _normalize_newlines(replace)

    if not find_nl:
        return None, "none"

    if before_nl.count(find_nl) == 1:
        after_nl = before_nl.replace(find_nl, replace_nl, replace_count)
        if after_nl != before_nl:
            return after_nl, "normalized_newlines"
        return before, "normalized_newlines_noop"

    pattern = _compile_whitespace_normalized_pattern(find_nl)
    matches = list(pattern.finditer(before_nl))
    if len(matches) == 1:
        match = matches[0]
        after_nl = before_nl[: match.start()] + replace_nl + before_nl[match.end() :]
        if after_nl != before_nl:
            return after_nl, "normalized_whitespace"
        return before, "normalized_whitespace_noop"

    # Fuzzy recovery is intentionally restricted to single targeted replacements.
    if count == 1:
        after_fuzzy = _apply_fuzzy_block_replacement(
            before_nl=before_nl,
            find_nl=find_nl,
            replace_nl=replace_nl,
        )
        if after_fuzzy is not None:
            if after_fuzzy != before_nl:
                return after_fuzzy, "fuzzy_block"
            return before, "fuzzy_block_noop"

    return None, "none"


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _compile_whitespace_normalized_pattern(text: str) -> re.Pattern[str]:
    pieces: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in {" ", "\t"}:
            while i < len(text) and text[i] in {" ", "\t"}:
                i += 1
            # Before a line break the newline piece absorbs the run: adjacent
            # [ \t]+[ \t]* classes backtracked polynomially (LLM-12).
            if i < len(text) and text[i] == "\n":
                continue
            pieces.append(r"[ \t]+")
            continue
        if ch == "\n":
            # Ignore trailing horizontal whitespace before line breaks.
            pieces.append(r"[ \t]*\n")
            i += 1
            continue
        pieces.append(re.escape(ch))
        i += 1
    return re.compile("".join(pieces))


_FUZZY_BLOCK_TIME_BUDGET_SEC = 2.0
"""Wall-time cap on one fuzzy block search (LLM-12)."""


def _apply_fuzzy_block_replacement(
    *,
    before_nl: str,
    find_nl: str,
    replace_nl: str,
) -> str | None:
    find_block = find_nl.strip()
    if not find_block:
        return None
    # Keep fuzzy replacement conservative to avoid accidental rewrites.
    if len(find_block) < 16:
        return None

    line_chunks = before_nl.splitlines(keepends=True)
    if not line_chunks:
        return None
    if len(line_chunks) > 6000:
        return None

    target_lines = len(find_block.splitlines())
    if target_lines <= 0:
        return None

    candidate_window_lengths = {target_lines}
    if target_lines > 1:
        candidate_window_lengths.add(target_lines - 1)
        candidate_window_lengths.add(target_lines + 1)
    candidate_window_lengths = {
        size for size in candidate_window_lengths if 1 <= size <= len(line_chunks)
    }
    if not candidate_window_lengths:
        return None

    cumulative_lengths = [0]
    for chunk in line_chunks:
        cumulative_lengths.append(cumulative_lengths[-1] + len(chunk))

    min_ratio = 0.96 if target_lines == 1 else 0.92
    # LLM-12: a window whose ratio can't reach min_ratio - margin can be neither the
    # accepted match nor within the ambiguity margin of one, so the cheap upper bounds
    # (real_quick_ratio, quick_ratio) skip it without changing the outcome. The matcher
    # caches the find block as its second sequence, and a wall-time cap makes a
    # pathological file fail closed instead of burning CPU outside any deadline.
    ambiguity_margin = 0.03
    prune_below = min_ratio - ambiguity_margin
    matcher = difflib.SequenceMatcher(None, "", find_block)
    deadline = time.monotonic() + _FUZZY_BLOCK_TIME_BUDGET_SEC
    scored_windows: list[tuple[float, int, int]] = []
    for window_len in sorted(candidate_window_lengths):
        max_start = len(line_chunks) - window_len
        for start in range(max_start + 1):
            if time.monotonic() > deadline:
                return None
            raw_window = "".join(line_chunks[start : start + window_len]).strip()
            if not raw_window:
                continue
            matcher.set_seq1(raw_window)
            if matcher.real_quick_ratio() < prune_below or matcher.quick_ratio() < prune_below:
                continue
            scored_windows.append((matcher.ratio(), start, window_len))

    if not scored_windows:
        return None

    scored_windows.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_start, best_len = scored_windows[0]
    second_ratio = scored_windows[1][0] if len(scored_windows) > 1 else 0.0

    if best_ratio < min_ratio:
        return None
    if (best_ratio - second_ratio) < ambiguity_margin:
        return None

    start_offset = cumulative_lengths[best_start]
    end_offset = cumulative_lengths[best_start + best_len]
    return before_nl[:start_offset] + replace_nl + before_nl[end_offset:]


def _normalize_full_file_response(raw: dict[str, Any]) -> tuple[str | None, PatcherMeta]:
    content = _get_first_choice_content(raw)
    if content is None:
        return None, {"error": "empty_full_file_response", "raw_response": _safe_trunc(raw)}

    repaired, code_fence_stripped = _extract_first_fenced_code_block(content)
    meta: PatcherMeta = {
        "full_file_code_fence_stripped": code_fence_stripped,
        "full_file_output_chars": len(repaired),
    }
    finish_reason = _get_first_choice_finish_reason(raw)
    if finish_reason:
        meta["finish_reason"] = finish_reason
    if finish_reason == "length":
        meta["error"] = "truncated_full_file_response"
        meta["raw_response"] = content[:2000]
        return None, meta
    if not repaired.strip():
        meta["error"] = "empty_full_file_response"
        meta["raw_response"] = content[:2000]
        return None, meta
    if _looks_like_raw_unified_diff(repaired):
        meta["error"] = "returned_diff_instead_of_file"
        meta["raw_response"] = repaired[:2000]
        return None, meta
    return repaired, meta


def _resolve_full_file_target(*, cwd: Path) -> tuple[str | None, str]:
    exclude_dirs = {".llmdebug", "__pycache__", ".venv", ".git"}
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(cwd):
        dirnames[:] = sorted(name for name in dirnames if name not in exclude_dirs)
        dirpath_path = Path(dirpath)
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = dirpath_path / filename
            if not path.is_file():
                continue
            rel = str(path.relative_to(cwd))
            if _is_test_file_path(rel):
                continue
            candidates.append(rel)
    candidates.sort()
    if len(candidates) == 1:
        return candidates[0], "single_non_test_python_file"
    if not candidates:
        return None, "no_non_test_python_files"
    return None, "multiple_non_test_python_files"


def _build_forced_attempt_diff(*, prompt: str, cwd: Path) -> tuple[str, str] | None:
    """Build a deterministic minimal patch when model output is empty/invalid."""
    candidates = _extract_file_index_paths(prompt)
    if not candidates:
        return None

    target_rel: str | None = None
    for rel in candidates:
        if not rel.endswith(".py"):
            continue
        lower = rel.lower()
        name = Path(rel).name.lower()
        if name.startswith("test_") or "/test" in lower or "\\test" in lower:
            continue
        target_rel = rel
        break
    if target_rel is None:
        for rel in candidates:
            if rel.endswith(".py"):
                target_rel = rel
                break
    if target_rel is None:
        return None

    target_path = (cwd / target_rel).resolve()
    try:
        target_path.relative_to(cwd.resolve())
    except Exception:
        return None
    if not target_path.exists():
        return None

    before = target_path.read_text(encoding="utf-8")
    marker = FORCED_PATCH_MARKER
    if marker in before:
        marker = f"{FORCED_PATCH_MARKER}-2"
    sep = "" if before.endswith("\n") else "\n"
    after = before + f"{sep}{marker}\n"
    if after == before:
        return None

    diff = _build_diff_from_changes([(target_rel, before, after)])
    return target_rel, diff


def _extract_file_index_paths(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Available files"):
            start = i + 1
            break
    if start is None:
        return []

    out: list[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            if out:
                break
            continue
        if s.endswith(":"):
            break
        if s.startswith("Pytest output") or s.startswith("Project files:"):
            break
        out.append(s)
    return out
