"""Diff/patch application, sanitization, syntax preflight, and relevance scoring.

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import difflib
import py_compile
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .eval_patch_constants import FORCED_PATCH_MARKER, PATCH_APPLY_TIMEOUT_SEC


@dataclass(frozen=True)
class _ParsedDiffEntry:
    """Normalized view of one file-level diff entry."""

    old_path: str | None
    new_path: str | None
    diff_old_path: str | None = None
    diff_new_path: str | None = None

    def real_paths(self) -> list[str]:
        out: list[str] = []
        for path in (self.old_path, self.new_path, self.diff_old_path, self.diff_new_path):
            if path is None or path == "/dev/null":
                continue
            normalized = str(PurePosixPath(path.replace("\\", "/")))
            if normalized not in out:
                out.append(normalized)
        return out


def _diff_lines(text: str) -> list[str]:
    """Split on ``\\n`` or ``\\r\\n`` only (LLM-15).

    ``str.splitlines`` also splits on form feeds, ``\\x1c``-``\\x1e``, ``\\x85`` and
    U+2028, which can appear inside source lines.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


RELAXED_APPLY_MESSAGE = "applied via relaxed replacement fallback"
_PATCH_LEFTOVER_SUFFIXES = (".rej", ".orig")


def patch_apply_mode(*, apply_ok: bool, apply_msg: str) -> str | None:
    """``strict`` or ``relaxed_fallback`` for an applied diff, ``None`` otherwise (ATT-10)."""
    if not apply_ok:
        return None
    return "relaxed_fallback" if apply_msg == RELAXED_APPLY_MESSAGE else "strict"


def _patch_leftovers(cwd: Path) -> set[Path]:
    return {
        path
        for suffix in _PATCH_LEFTOVER_SUFFIXES
        for path in cwd.rglob(f"*{suffix}")
        if path.is_file() and not path.is_symlink()
    }


def _remove_new_patch_leftovers(cwd: Path, before: set[Path], *, diff: str) -> None:
    """Delete ``.rej``/``.orig`` files that ``patch`` created (ATT-16).

    A file the diff itself names is an intended target, not a tool artefact, even with
    one of those suffixes, so cleanup leaves it alone.
    """
    intended = {(cwd / rel).resolve() for rel in _extract_diff_paths(diff)}
    for path in _patch_leftovers(cwd) - before:
        if path.resolve() in intended:
            continue
        path.unlink(missing_ok=True)


def _crlf_diff_targets(cwd: Path, diff: str) -> list[Path]:
    """Existing diff targets whose every line ends in CRLF (LLM-09)."""
    targets: list[Path] = []
    for rel in _extract_diff_paths(diff):
        target = cwd / rel
        if target.is_symlink() or not target.is_file():
            continue
        data = target.read_bytes()
        if b"\r\n" in data and data.count(b"\r\n") == data.count(b"\n"):
            targets.append(target)
    return targets


def apply_diff(cwd: Path, diff: str) -> tuple[bool, str]:
    """Apply a unified diff to *cwd* via ``patch`` or ``git apply``.

    Falls back to relaxed line-by-line replacement when strict patch
    application fails. Files that use CRLF line endings are patched as LF and
    converted back afterwards, so a model diff (always LF) matches them and the
    file keeps its endings (LLM-09).

    Returns:
        Tuple of (success, error_or_info_message).
    """
    path_ok, path_error = _validate_diff_paths_within_cwd(cwd, diff)
    if not path_ok:
        return False, path_error
    structure_ok, structure_error = _validate_unified_diff_structure(diff)
    if not structure_ok:
        return False, structure_error

    crlf_targets = _crlf_diff_targets(cwd, diff)
    if not crlf_targets:
        return _apply_validated_diff(cwd, diff)
    normalized: list[Path] = []
    try:
        for target in crlf_targets:
            # Preparing one target can fail (a read-only file) after earlier ones were
            # already rewritten, so the rollback has to cover those too (LLM-09).
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
            normalized.append(target)
        return _apply_validated_diff(cwd, diff.replace("\r\n", "\n"))
    except OSError as exc:
        return False, f"could not prepare CRLF diff targets: {type(exc).__name__}: {exc}"
    finally:
        for target in normalized:
            if target.is_file() and not target.is_symlink():
                target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))


def _apply_validated_diff(cwd: Path, diff: str) -> tuple[bool, str]:
    # Prefer POSIX patch so paths are resolved relative to `cwd` (not git toplevel).
    patch = shutil.which("patch")
    if patch:
        strict_backups = _backup_diff_target_files(cwd=cwd, diff=diff)
        leftovers_before = _patch_leftovers(cwd)
        try:
            proc = subprocess.run(
                [patch, "-p1", "--batch", "--forward"],
                input=diff,
                text=True,
                cwd=cwd,
                capture_output=True,
                timeout=PATCH_APPLY_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            _remove_new_patch_leftovers(cwd, leftovers_before, diff=diff)
            return False, f"patch timed out after {PATCH_APPLY_TIMEOUT_SEC}s"
        _remove_new_patch_leftovers(cwd, leftovers_before, diff=diff)
        if proc.returncode == 0:
            _remove_deleted_diff_targets(cwd=cwd, diff=diff)
            return True, ""
        msg = (proc.stderr or proc.stdout or "").strip()
        _restore_diff_target_files(cwd=cwd, backups=strict_backups)
        relaxed_ok, relaxed_msg = _apply_relaxed_replacements(cwd, diff)
        if relaxed_ok:
            return True, RELAXED_APPLY_MESSAGE
        return False, f"patch failed: {msg}; relaxed fallback failed: {relaxed_msg}"

    # Fallback to git apply when patch utility is unavailable.
    git = shutil.which("git")
    if git:
        try:
            proc = subprocess.run(
                [git, "apply", "--whitespace=nowarn", "-"],
                input=diff,
                text=True,
                cwd=cwd,
                capture_output=True,
                timeout=PATCH_APPLY_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return False, f"git apply timed out after {PATCH_APPLY_TIMEOUT_SEC}s"
        if proc.returncode == 0:
            _remove_deleted_diff_targets(cwd=cwd, diff=diff)
            return True, ""
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, f"git apply failed: {msg}"

    return False, "No patch tool found (git or patch required)."


def _extract_diff_paths(diff: str) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for entry in _parse_diff_entries(diff):
        for path in entry.real_paths():
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
    return unique


def _parse_diff_entries(diff: str) -> list[_ParsedDiffEntry]:
    entries: list[_ParsedDiffEntry] = []
    diff_old_path: str | None = None
    diff_new_path: str | None = None
    pending_old_path: str | None = None

    for raw_line in _diff_lines(diff):
        stripped = raw_line.strip()
        if stripped.startswith("diff --git "):
            tokens = stripped.split()
            diff_old_path = _normalize_diff_path_token(tokens[2]) if len(tokens) >= 3 else None
            diff_new_path = _normalize_diff_path_token(tokens[3]) if len(tokens) >= 4 else None
            pending_old_path = None
            continue
        if raw_line.startswith("--- "):
            token = raw_line[4:].strip().split()[0] if raw_line[4:].strip() else ""
            pending_old_path = _normalize_diff_path_token(token)
            continue
        if raw_line.startswith("+++ "):
            token = raw_line[4:].strip().split()[0] if raw_line[4:].strip() else ""
            entries.append(
                _ParsedDiffEntry(
                    old_path=pending_old_path,
                    new_path=_normalize_diff_path_token(token),
                    diff_old_path=diff_old_path,
                    diff_new_path=diff_new_path,
                )
            )
            pending_old_path = None
            diff_old_path = None
            diff_new_path = None
    return entries


def _normalize_diff_path_token(token: str) -> str | None:
    if not token:
        return None
    if token in {"a/", "b/"}:
        return None
    if token.startswith("a/") or token.startswith("b/"):
        token = token[2:]
    return token


def _is_test_file_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    parts = [part.lower() for part in pure.parts]
    if not parts:
        return False
    file_name = parts[-1]
    if any(part in {"tests", "test"} for part in parts):
        return True
    if file_name == "conftest.py":
        return True
    if file_name.startswith("test_"):
        return True
    return bool(file_name.endswith("_test.py"))


def _compute_patch_path_metrics(diff: str) -> dict[str, Any]:
    normalized_paths = [
        str(PurePosixPath(path.replace("\\", "/"))) for path in _extract_diff_paths(diff)
    ]
    test_paths = [path for path in normalized_paths if _is_test_file_path(path)]
    non_test_paths = [path for path in normalized_paths if not _is_test_file_path(path)]
    return {
        "patch_file_count": len(normalized_paths),
        "patch_test_file_count": len(test_paths),
        "patch_non_test_file_count": len(non_test_paths),
        "patch_touches_tests": bool(test_paths),
        "patch_file_paths": normalized_paths,
        "patch_test_file_paths": test_paths,
    }


def _backup_diff_target_files(*, cwd: Path, diff: str) -> dict[str, bytes]:
    backups: dict[str, bytes] = {}
    for entry in _parse_diff_entries(diff):
        for rel in entry.real_paths():
            target = cwd / rel
            if not target.exists() or not target.is_file():
                continue
            if rel in backups:
                continue
            try:
                backups[rel] = target.read_bytes()
            except Exception:
                continue
    return backups


def _deleted_diff_target_paths(diff: str) -> list[str]:
    deleted_paths: list[str] = []
    for entry in _parse_diff_entries(diff):
        if entry.new_path != "/dev/null" or entry.old_path is None or entry.old_path == "/dev/null":
            continue
        normalized = str(PurePosixPath(entry.old_path.replace("\\", "/")))
        if normalized not in deleted_paths:
            deleted_paths.append(normalized)
    return deleted_paths


def _remove_deleted_diff_targets(*, cwd: Path, diff: str) -> None:
    for rel in _deleted_diff_target_paths(diff):
        target = cwd / rel
        try:
            if target.exists():
                target.unlink()
        except Exception:
            continue


def _restore_diff_target_files(*, cwd: Path, backups: dict[str, bytes]) -> None:
    for rel, payload in backups.items():
        target = cwd / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        except Exception:
            continue


def _python_files_touched_by_diff(*, cwd: Path, diff: str) -> list[str]:
    out: list[str] = []
    for rel in _extract_diff_paths(diff):
        if not rel.endswith(".py"):
            continue
        target = cwd / rel
        if not target.exists() or not target.is_file():
            continue
        if rel not in out:
            out.append(rel)
    return out


def _patch_sanitize_line_reasons(line: str) -> list[str]:
    reasons: list[str] = []
    if "\\Exception" in line:
        reasons.append("escaped_backslash_before_exception_token")
    if "\\n" in line or "\\t" in line:
        contains_quoted_escape = bool(re.search(r"""['"][^'"]*(\\n|\\t)[^'"]*['"]""", line))
        if not contains_quoted_escape:
            reasons.append("literal_escaped_newline_or_tab_in_code")
    if "await " in line and line.rstrip().endswith('",'):
        reasons.append("suspicious_trailing_quote_comma_after_await")
    return reasons


def _run_patch_sanitization(*, diff: str) -> tuple[bool, dict[str, Any]]:
    """Scan added lines in a diff for malformed patch artifacts.

    Returns (passed, details_dict) where *passed* is ``False`` when an added line in a
    Python file has one of the artifacts ``_patch_sanitize_line_reasons`` checks for: a
    backslash before ``Exception``, a literal ``\\n``/``\\t`` outside a string literal,
    or an ``await`` line ending in ``",``. Raw diff headers and markdown fences inside
    added lines are not checked (ATT-19).
    """
    current_file = ""
    current_new_lineno = 0
    in_hunk = False
    checked_files: list[str] = []
    findings: list[str] = []

    for raw_line in _diff_lines(diff):
        if raw_line.startswith("+++ "):
            token = raw_line[4:].strip().split()[0] if raw_line[4:].strip() else ""
            normalized = _normalize_diff_path_token(token)
            current_file = str(PurePosixPath((normalized or "").replace("\\", "/")))
            in_hunk = False
            if current_file and current_file.endswith(".py") and current_file not in checked_files:
                checked_files.append(current_file)
            continue
        if raw_line.startswith("@@"):
            match = re.search(r"\+(\d+)", raw_line)
            current_new_lineno = int(match.group(1)) - 1 if match else 0
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if not current_file or not current_file.endswith(".py"):
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++ "):
            current_new_lineno += 1
            line = raw_line[1:]
            line_reasons = _patch_sanitize_line_reasons(line)
            findings.extend(
                f"{current_file}:{current_new_lineno}:{reason}" for reason in line_reasons
            )
            continue
        if raw_line.startswith(" "):
            current_new_lineno += 1
            continue
        if raw_line.startswith("-"):
            continue

    if not findings:
        return True, {
            "patch_sanitize_checked_files": checked_files,
            "patch_sanitize_checked_count": len(checked_files),
            "patch_sanitize_reasons": [],
            "patch_sanitize_error_msg": "",
        }

    preview = ", ".join(findings[:6])
    if len(findings) > 6:
        preview = f"{preview}, ... (+{len(findings) - 6} more)"
    return False, {
        "patch_sanitize_checked_files": checked_files,
        "patch_sanitize_checked_count": len(checked_files),
        "patch_sanitize_reasons": findings,
        "patch_sanitize_error_msg": f"Patch sanitation rejected malformed patch text: {preview}",
    }


def _run_syntax_preflight(*, cwd: Path, diff: str) -> tuple[bool, dict[str, Any]]:
    """Compile-check all Python files touched by *diff*.

    Returns (passed, details_dict) where *passed* is ``False`` when
    any file fails ``py_compile``.
    """
    python_files = _python_files_touched_by_diff(cwd=cwd, diff=diff)
    if not python_files:
        return True, {
            "syntax_preflight_checked_files": [],
            "syntax_preflight_checked_count": 0,
            "syntax_preflight_error_file": "",
            "syntax_preflight_error_type": "",
            "syntax_preflight_error_msg": "",
        }

    with tempfile.TemporaryDirectory(prefix="llmdebug-syntax-preflight-") as tmpdir:
        tmp_root = Path(tmpdir)
        for idx, rel in enumerate(python_files):
            source_path = cwd / rel
            pyc_path = tmp_root / f"{idx:04d}.pyc"
            try:
                py_compile.compile(
                    str(source_path),
                    cfile=str(pyc_path),
                    dfile=rel,
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                return False, {
                    "syntax_preflight_checked_files": list(python_files),
                    "syntax_preflight_checked_count": len(python_files),
                    "syntax_preflight_error_file": rel,
                    "syntax_preflight_error_type": type(exc).__name__,
                    "syntax_preflight_error_msg": str(exc).strip(),
                }
            except Exception as exc:
                return False, {
                    "syntax_preflight_checked_files": list(python_files),
                    "syntax_preflight_checked_count": len(python_files),
                    "syntax_preflight_error_file": rel,
                    "syntax_preflight_error_type": type(exc).__name__,
                    "syntax_preflight_error_msg": f"{type(exc).__name__}: {exc}",
                }

    return True, {
        "syntax_preflight_checked_files": list(python_files),
        "syntax_preflight_checked_count": len(python_files),
        "syntax_preflight_error_file": "",
        "syntax_preflight_error_type": "",
        "syntax_preflight_error_msg": "",
    }


def _render_syntax_preflight_retry_feedback(
    *,
    cycle: int,
    max_cycles: int,
    error_file: str,
    error_type: str,
    error_msg: str,
) -> str:
    trimmed_error = (error_msg or "").strip()
    if len(trimmed_error) > 800:
        trimmed_error = trimmed_error[:797] + "..."
    return (
        "\n\n"
        "Retry request: prior patch introduced a Python syntax/compile error.\n"
        f"- cycle: {cycle}/{max_cycles}\n"
        f"- file: {error_file or 'unknown'}\n"
        f"- error_type: {error_type or 'SyntaxError'}\n"
        f"- error: {trimmed_error or 'unknown syntax error'}\n"
        "Please return a corrected unified diff that fixes this syntax issue first.\n"
    )


def _render_patch_sanitize_retry_feedback(
    *,
    cycle: int,
    max_cycles: int,
    sanitize_error: str,
) -> str:
    trimmed_error = (sanitize_error or "").strip()
    if len(trimmed_error) > 800:
        trimmed_error = trimmed_error[:797] + "..."
    return (
        "\n\n"
        "Retry request: prior patch text was malformed and rejected by sanitation checks.\n"
        f"- cycle: {cycle}/{max_cycles}\n"
        f"- issue: {trimmed_error or 'unknown patch sanitation issue'}\n"
        "Please return a corrected unified diff without escaped newline/tab artifacts.\n"
    )


def _render_patch_verifier_recovery_feedback(
    *,
    score: int | None,
    min_score: int,
    reasons: Sequence[str],
    patch_files: Sequence[str],
    crash_file: str,
    inferred_source_scope: Sequence[str],
) -> str:
    trimmed_reasons = (
        ", ".join(str(item) for item in reasons[:6]) or "insufficient_source_relevance"
    )
    if len(trimmed_reasons) > 400:
        trimmed_reasons = trimmed_reasons[:397] + "..."
    trimmed_patch_files = ", ".join(str(item) for item in patch_files[:6]) or "none"
    if len(trimmed_patch_files) > 300:
        trimmed_patch_files = trimmed_patch_files[:297] + "..."
    trimmed_scope = ", ".join(str(item) for item in inferred_source_scope[:6]) or "none"
    if len(trimmed_scope) > 300:
        trimmed_scope = trimmed_scope[:297] + "..."
    score_text = str(score) if isinstance(score, int) else "unknown"
    return (
        "\n"
        "Verifier-guided repair request: the previous candidate was substantive but failed the "
        "pre-apply source-relevance verifier.\n"
        f"- score: {score_text}/{int(min_score)}\n"
        f"- reasons: {trimmed_reasons}\n"
        f"- touched_files: {trimmed_patch_files}\n"
        f"- crash_file: {crash_file or 'unknown'}\n"
        f"- inferred_source_scope: {trimmed_scope}\n"
        "Return one replacement unified diff that moves toward traceback-linked non-test source "
        "edits first. Do not explain the patch; only return the diff.\n"
    )


def _extract_candidate_patch_files(diff: str) -> list[str]:
    return [str(PurePosixPath(path.replace("\\", "/"))) for path in _extract_diff_paths(diff)]


def _patch_mentions_crash_file(*, patch_files: list[str], crash_file: str | None) -> bool:
    if not crash_file:
        return False
    crash_norm = str(PurePosixPath(crash_file.replace("\\", "/")))
    crash_name = PurePosixPath(crash_norm).name
    for path in patch_files:
        path_norm = str(PurePosixPath(path.replace("\\", "/")))
        if path_norm == crash_norm or PurePosixPath(path_norm).name == crash_name:
            return True
    return False


def _patch_mentions_failing_test_context(
    *, patch_files: list[str], failing_test_file: str | None
) -> bool:
    if not failing_test_file:
        return False
    test_norm = str(PurePosixPath(failing_test_file.replace("\\", "/")))
    test_name = PurePosixPath(test_norm).name
    for path in patch_files:
        path_norm = str(PurePosixPath(path.replace("\\", "/")))
        if path_norm == test_norm or PurePosixPath(path_norm).name == test_name:
            return True
    return False


def _patch_mentions_scope_files(*, patch_files: list[str], scope_files: set[str]) -> bool:
    if not scope_files:
        return False
    scope_by_name: dict[str, set[str]] = {}
    for scope_path in scope_files:
        scope_name = PurePosixPath(scope_path).name
        scope_by_name.setdefault(scope_name, set()).add(scope_path)
    for path in patch_files:
        path_norm = str(PurePosixPath(path.replace("\\", "/")))
        if path_norm in scope_files:
            return True
        # Compatibility fallback: allow basename-only matches only when the patch path
        # itself has no directory segments and maps unambiguously to exactly one scope file.
        if "/" not in path_norm:
            candidates = scope_by_name.get(PurePosixPath(path_norm).name, set())
            if len(candidates) == 1:
                return True
    return False


def _patch_contains_substantive_change(diff: str) -> bool:
    for line in _diff_lines(diff):
        if not line:
            continue
        if line.startswith(("diff --git ", "index ", "@@", "--- ", "+++ ")):
            continue
        if line.startswith(("+", "-")):
            return True
    return False


def _is_safe_relative_path(path: str) -> bool:
    if not path or path == "/dev/null":
        return False
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute():
        return False
    parts = pure.parts
    if not parts:
        return False
    return not any(part in {"", ".", ".."} for part in parts)


def _validate_diff_paths_within_cwd(cwd: Path, diff: str) -> tuple[bool, str]:
    entries = _parse_diff_entries(diff)
    if not entries:
        return False, "patch rejected: no file paths found in diff"
    root = cwd.resolve()
    for entry in entries:
        if entry.old_path == "/dev/null" and entry.new_path == "/dev/null":
            return False, "patch rejected: invalid file header uses /dev/null on both sides"
        for source, raw_path in (
            ("diff --git", entry.diff_old_path),
            ("diff --git", entry.diff_new_path),
            ("file header", entry.old_path),
            ("file header", entry.new_path),
        ):
            if raw_path is None:
                continue
            if raw_path == "/dev/null":
                if source == "diff --git":
                    return (
                        False,
                        "patch rejected: /dev/null is only supported in ---/+++ headers",
                    )
                continue
            if not _is_safe_relative_path(raw_path):
                return False, f"patch rejected: unsafe path {raw_path!r}"
            resolved = (root / raw_path).resolve()
            try:
                resolved.relative_to(root)
            except Exception:
                return False, f"patch rejected: path escapes workdir {raw_path!r}"
    return True, ""


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def _validate_unified_diff_structure(diff: str) -> tuple[bool, str]:
    expected_old = 0
    expected_new = 0
    actual_old = 0
    actual_new = 0
    in_hunk = False

    def _counts_match() -> bool:
        return actual_old == expected_old and actual_new == expected_new

    for raw_line in _diff_lines(diff):
        if raw_line.startswith("@@"):
            if in_hunk and not _counts_match():
                return False, "patch rejected: malformed unified diff hunk counts"
            match = _HUNK_HEADER_RE.match(raw_line)
            if match is None:
                return False, "patch rejected: malformed unified diff hunk header"
            expected_old = int(match.group("old_count") or "1")
            expected_new = int(match.group("new_count") or "1")
            actual_old = 0
            actual_new = 0
            in_hunk = True
            continue

        if raw_line.startswith(("diff --git ", "--- ", "+++ ")):
            if in_hunk and not _counts_match():
                return False, "patch rejected: malformed unified diff hunk counts"
            in_hunk = False
            continue

        if not in_hunk:
            continue
        if raw_line.startswith("\\"):
            continue
        if raw_line.startswith(" "):
            actual_old += 1
            actual_new += 1
            continue
        if raw_line.startswith("-"):
            actual_old += 1
            continue
        if raw_line.startswith("+"):
            actual_new += 1
            continue
        return False, "patch rejected: malformed unified diff hunk content"

    if in_hunk and not _counts_match():
        return False, "patch rejected: malformed unified diff hunk counts"
    return True, ""


def _apply_relaxed_replacements(cwd: Path, diff: str) -> tuple[bool, str]:
    """Best-effort fallback for near-correct model diffs with bad hunk context.

    Supports only line-for-line replacements extracted from unified hunks.
    """
    replacements: dict[str, list[tuple[list[str], list[str], list[str], list[str]]]] = {}
    current_path: str | None = None
    lines = _diff_lines(diff)
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        if raw_line.startswith("+++ b/"):
            current_path = raw_line[6:].strip()
            i += 1
            continue
        if raw_line.startswith("@@"):
            if current_path is None:
                i += 1
                continue
            hunk_lines: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if (
                    nxt.startswith("@@")
                    or nxt.startswith("diff --git ")
                    or nxt.startswith("--- a/")
                    or nxt.startswith("+++ b/")
                ):
                    break
                hunk_lines.append(nxt)
                i += 1
            replacements.setdefault(current_path, []).extend(
                _extract_relaxed_pairs_from_hunk(hunk_lines)
            )
            continue
        i += 1

    if not replacements:
        return False, "no replaceable relaxed hunk blocks found"

    root = cwd.resolve()
    pending_updates: list[tuple[Path, str]] = []
    for rel_path, pairs in replacements.items():
        file_path = (cwd / rel_path).resolve()
        try:
            file_path.relative_to(root)
        except Exception:
            return False, f"path outside working directory: {rel_path}"
        if not file_path.exists():
            return False, f"file not found: {rel_path}"
        text = file_path.read_text(encoding="utf-8")
        had_newline = text.endswith("\n")
        line_values = _diff_lines(text)

        changed = False
        for old_lines, new_lines, before_ctx, after_ctx in pairs:
            matching_indices: list[int] = []
            if old_lines:
                max_start = len(line_values) - len(old_lines) + 1
                for idx in range(max(0, max_start)):
                    if line_values[idx : idx + len(old_lines)] != old_lines:
                        continue
                    if before_ctx:
                        start = idx - len(before_ctx)
                        if start < 0 or line_values[start:idx] != before_ctx:
                            continue
                    if after_ctx:
                        end = idx + len(old_lines) + len(after_ctx)
                        if (
                            end > len(line_values)
                            or line_values[idx + len(old_lines) : end] != after_ctx
                        ):
                            continue
                    matching_indices.append(idx)
            else:
                for idx in range(len(line_values) + 1):
                    if before_ctx:
                        start = idx - len(before_ctx)
                        if start < 0 or line_values[start:idx] != before_ctx:
                            continue
                    if after_ctx:
                        end = idx + len(after_ctx)
                        if end > len(line_values) or line_values[idx:end] != after_ctx:
                            continue
                    matching_indices.append(idx)

            if len(matching_indices) == 0:
                return (
                    False,
                    f"could not find context-anchored target block in {rel_path!r}",
                )
            if len(matching_indices) > 1:
                return False, f"ambiguous relaxed replacement target block in {rel_path!r}"
            start = matching_indices[0]
            end = start + len(old_lines)
            if line_values[start:end] != new_lines:
                line_values[start:end] = new_lines
                changed = True

        if changed:
            updated = "\n".join(line_values)
            if had_newline:
                updated += "\n"
            pending_updates.append((file_path, updated))

    if not pending_updates:
        return False, "relaxed replacement produced no file content changes"

    for file_path, updated in pending_updates:
        file_path.write_text(updated, encoding="utf-8")

    return True, ""


def _extract_relaxed_pairs_from_hunk(
    hunk_lines: list[str],
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    """Extract replaceable change blocks with nearby context anchors."""
    out: list[tuple[list[str], list[str], list[str], list[str]]] = []
    recent_context: list[str] = []
    i = 0
    while i < len(hunk_lines):
        line = hunk_lines[i]
        if line.startswith("\\"):
            i += 1
            continue
        if line.startswith(" "):
            recent_context.append(line[1:])
            if len(recent_context) > 3:
                recent_context.pop(0)
            i += 1
            continue
        if line.startswith(("-", "+")):
            before_ctx = recent_context[-2:] if recent_context else []
            old_lines: list[str] = []
            new_lines: list[str] = []
            while i < len(hunk_lines):
                current = hunk_lines[i]
                if current.startswith("\\"):
                    i += 1
                    continue
                if current.startswith("-"):
                    old_lines.append(current[1:])
                    i += 1
                    continue
                if current.startswith("+"):
                    new_lines.append(current[1:])
                    i += 1
                    continue
                break
            after_ctx: list[str] = []
            j = i
            while j < len(hunk_lines):
                current = hunk_lines[j]
                if current.startswith("\\"):
                    j += 1
                    continue
                if current.startswith(" ") and len(after_ctx) < 2:
                    after_ctx.append(current[1:])
                    j += 1
                    continue
                break
            out.append((old_lines, new_lines, before_ctx, after_ctx))
            continue
        i += 1
    return out


def _is_forced_noop_marker_diff(diff: str) -> bool:
    changed_lines_seen = False
    for line in _diff_lines(str(diff or "")):
        if not line:
            continue
        if line.startswith(
            (
                "diff --git ",
                "index ",
                "new file mode ",
                "deleted file mode ",
                "old mode ",
                "new mode ",
                "similarity index ",
                "rename from ",
                "rename to ",
                "copy from ",
                "copy to ",
                "--- ",
                "+++ ",
                "@@",
                "\\",
            )
        ):
            continue
        if line.startswith(" "):
            continue
        if line.startswith("+") or line.startswith("-"):
            payload = line[1:].strip()
            if not payload:
                continue
            changed_lines_seen = True
            if not payload.startswith(FORCED_PATCH_MARKER):
                return False
            continue
        return False
    return changed_lines_seen


def _build_runner_forced_attempt_diff(
    *, cwd: Path, file_index: str, attempt: int
) -> tuple[str, str] | None:
    """Build a deterministic minimal patch when a patcher returns no diff."""
    candidates = [line.strip() for line in file_index.splitlines() if line.strip()]
    if not candidates:
        return None

    target_rel = _select_forced_patch_target(candidates)
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
    marker_base = f"{FORCED_PATCH_MARKER}-{attempt}"
    marker = marker_base
    suffix = 2
    while marker in before:
        marker = f"{marker_base}-{suffix}"
        suffix += 1
    sep = "" if before.endswith("\n") else "\n"
    after = before + f"{sep}{marker}\n"
    if after == before:
        return None

    diff = _build_diff_from_changes([(target_rel, before, after)])
    if not diff.strip():
        return None
    return target_rel, diff


def _select_forced_patch_target(candidates: list[str]) -> str | None:
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
    if target_rel is not None:
        return target_rel
    for rel in candidates:
        if rel.endswith(".py"):
            return rel
    return None


_NO_NEWLINE_MARKER = "\\ No newline at end of file\n"


def _keepends_lines(text: str) -> list[str]:
    """Split on ``\\n`` only, keeping terminators; a ``\\r`` stays part of its line."""
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def _build_diff_from_changes(changes: list[tuple[str, str, str]]) -> str:
    """Build a git-style diff from ``(path, before, after)`` texts.

    Lines are split on ``\\n`` only and a missing final newline gets the standard
    marker (LLM-15): ``str.splitlines`` also split on form feeds, ``\\x1c``-``\\x1e``,
    ``\\x85`` and U+2028, and dropped ``\\r``, so the diff didn't describe the file.
    """
    parts: list[str] = []
    for path, before, after in changes:
        file_lines = list(
            difflib.unified_diff(
                _keepends_lines(before),
                _keepends_lines(after),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="\n",
            )
        )
        if not file_lines:
            continue
        parts.append(f"diff --git a/{path} b/{path}\n")
        parts.extend(
            line if line.endswith("\n") else f"{line}\n{_NO_NEWLINE_MARKER}" for line in file_lines
        )
    return "".join(parts)
