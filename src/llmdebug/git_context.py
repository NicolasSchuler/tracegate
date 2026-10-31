"""Git repository context capture for debug snapshots."""

from __future__ import annotations

import re
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, cast

from .config import GitDirtyMode


class _GitError:
    """Sentinel to distinguish 'command failed' from 'not applicable'.

    Attributes:
        reason: Short description of the failure, such as ``"timeout"``.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclass(frozen=True)
class _GitContextCacheEntry:
    """Small in-process cache for git context lookups.

    Attributes:
        expires_at: ``time.monotonic()`` deadline after which the entry is stale.
        cwd: Working directory when the entry was stored.
        repo_root: Repository root, or None if the cwd was not in a repository.
        git_dir: Resolved git directory, used to detect a changed repository.
        head_mtime_ns: Modification time of ``HEAD`` when stored.
        index_mtime_ns: Modification time of the index when stored.
        dirty_cached: Dirty status when stored, reused by ``dirty_mode="cached"``.
        context_static: The context without its ``dirty`` key, or None.
    """

    expires_at: float
    cwd: str | None
    repo_root: str | None
    git_dir: str | None
    head_mtime_ns: int | None
    index_mtime_ns: int | None
    dirty_cached: bool | None
    context_static: dict[str, Any] | None


_GIT_CONTEXT_CACHE_TTL_SEC = 1.0
_GIT_CONTEXT_CACHE_LOCK = Lock()
_git_context_cache: _GitContextCacheEntry | None = None


def _run_git_command(
    args: list[str],
    timeout: float = 2.0,
    repo_root: str | None = None,
) -> str | _GitError:
    """Run a git command with timeout.

    Returns:
        The stripped stdout on exit code 0; otherwise a ``_GitError`` naming the
        failure (non-zero exit, timeout, missing git, or another ``OSError``).
    """
    cmd = ["git"]
    if repo_root:
        cmd.extend(["-C", repo_root])
    cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            # Not the locale encoding with strict errors: a non-UTF-8 author name or
            # file path in git output raised UnicodeDecodeError past these handlers (RCA-14).
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return _GitError(f"exit code {result.returncode}")
    except subprocess.TimeoutExpired:
        return _GitError("timeout")
    except FileNotFoundError:
        return _GitError("git not found")
    except OSError as e:
        return _GitError(f"OSError: {e}")


def _run_git_command_rc(
    args: list[str],
    timeout: float = 2.0,
    repo_root: str | None = None,
) -> int | _GitError:
    """Run a git command and return its exit code, or a ``_GitError`` if it cannot run."""
    cmd = ["git"]
    if repo_root:
        cmd.extend(["-C", repo_root])
    cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        return _GitError("timeout")
    except FileNotFoundError:
        return _GitError("git not found")
    except OSError as e:
        return _GitError(f"OSError: {e}")


def _resolve_cwd() -> str | None:
    """Return the resolved current working directory, or None if it cannot be read."""
    try:
        return str(Path.cwd().resolve())
    except (OSError, RuntimeError, ValueError):
        return None


def _path_within_repo(path: str, repo_root: str) -> bool:
    """Return whether *path* resolves to a location inside *repo_root*."""
    try:
        Path(path).resolve().relative_to(Path(repo_root).resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _resolve_git_dir(repo_root: str) -> Path | None:
    """Return the git directory of *repo_root*, following a ``.git`` file's ``gitdir:``."""
    git_entry = Path(repo_root) / ".git"
    try:
        if git_entry.is_dir():
            return git_entry.resolve()
        if not git_entry.is_file():
            return None

        raw = git_entry.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return None
        prefix = "gitdir:"
        if not raw.lower().startswith(prefix):
            return None
        target_raw = raw[len(prefix) :].strip()
        if not target_raw:
            return None

        git_dir = Path(target_raw)
        if not git_dir.is_absolute():
            git_dir = (git_entry.parent / git_dir).resolve()
        else:
            git_dir = git_dir.resolve()
        return git_dir
    except (OSError, RuntimeError, ValueError):
        return None


def _git_signal_mtimes(repo_root: str) -> tuple[str | None, int | None, int | None]:
    """Return the git directory and the mtimes of its ``HEAD`` and ``index`` files.

    These signals invalidate the git context cache when the repository changes.
    """
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None, None, None

    def _mtime_ns(path: Path) -> int | None:
        """Return the mtime of *path* in nanoseconds, or None if it cannot be read."""
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    return (
        str(git_dir),
        _mtime_ns(git_dir / "HEAD"),
        _mtime_ns(git_dir / "index"),
    )


def _detect_untracked_status(repo_root: str) -> bool | None:
    """Return untracked-file status using `ls-files` exit codes."""
    untracked = _run_git_command_rc(
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "--directory",
            "--no-empty-directory",
            "--error-unmatch",
            "--",
            "*",
        ],
        repo_root=repo_root,
    )
    if isinstance(untracked, _GitError):
        return None
    if untracked == 0:
        return True
    if untracked == 1:
        return False
    return None


def _detect_dirty_status(repo_root: str) -> bool | None:
    """Return dirty status with a faster command pair than `status --porcelain`."""
    unstaged = _run_git_command_rc(
        ["diff-files", "--quiet", "--ignore-submodules", "--"],
        repo_root=repo_root,
    )
    if isinstance(unstaged, _GitError):
        return None
    if unstaged == 1:
        return True
    if unstaged != 0:
        return None

    staged = _run_git_command_rc(
        ["diff-index", "--quiet", "--cached", "HEAD", "--"],
        repo_root=repo_root,
    )
    if isinstance(staged, _GitError):
        return None
    if staged == 1:
        return True
    if staged == 0:
        return _detect_untracked_status(repo_root)
    return None


def _hydrate_context_with_dirty(
    context_static: dict[str, Any], *, repo_root: str
) -> dict[str, Any]:
    """Return a copy of *context_static* with a freshly detected ``dirty`` status."""
    context = deepcopy(context_static)

    dirty = _detect_dirty_status(repo_root)
    if dirty is None:
        status = _run_git_command(["status", "--porcelain"], repo_root=repo_root)
        if isinstance(status, str):
            dirty = len(status) > 0
    if dirty is not None:
        context["dirty"] = dirty
    return context


def _load_cached_git_context(*, dirty_mode: GitDirtyMode) -> tuple[bool, dict[str, Any] | None]:
    """Look up the cached git context for the current working directory.

    The entry is used only while its TTL lasts, the cwd stays in the same
    repository, and ``HEAD`` and the index are unchanged.

    Returns:
        ``(hit, context)``: whether the cache answered and, on a hit, the context
        (None for a non-repository directory) with ``dirty`` set per *dirty_mode*.
    """
    now = time.monotonic()
    with _GIT_CONTEXT_CACHE_LOCK:
        entry = _git_context_cache
    if entry is None or now >= entry.expires_at:
        return False, None

    cwd = _resolve_cwd()
    if entry.repo_root is None:
        if cwd != entry.cwd:
            return False, None
        return True, None

    if cwd is None or not _path_within_repo(cwd, entry.repo_root):
        return False, None

    git_dir, head_mtime_ns, index_mtime_ns = _git_signal_mtimes(entry.repo_root)
    if (
        git_dir != entry.git_dir
        or head_mtime_ns != entry.head_mtime_ns
        or index_mtime_ns != entry.index_mtime_ns
    ):
        return False, None

    if entry.context_static is None:
        return True, None

    if dirty_mode == "off":
        return True, deepcopy(entry.context_static)

    if dirty_mode == "cached":
        if entry.dirty_cached is None:
            return False, None
        context = deepcopy(entry.context_static)
        context["dirty"] = entry.dirty_cached
        return True, context

    return True, _hydrate_context_with_dirty(entry.context_static, repo_root=entry.repo_root)


def _store_git_context_cache(
    context: dict[str, Any] | None,
    *,
    repo_root: str | None,
) -> None:
    """Cache *context* for the current cwd, keeping its ``dirty`` value separately."""
    cwd = _resolve_cwd()
    git_dir, head_mtime_ns, index_mtime_ns = (None, None, None)
    dirty_cached: bool | None = None
    context_static: dict[str, Any] | None = None

    if repo_root is not None:
        git_dir, head_mtime_ns, index_mtime_ns = _git_signal_mtimes(repo_root)
        if context:
            context_static = deepcopy(context)
            dirty_value = context_static.pop("dirty", None)
            if isinstance(dirty_value, bool):
                dirty_cached = dirty_value

    entry = _GitContextCacheEntry(
        expires_at=time.monotonic() + max(0.0, _GIT_CONTEXT_CACHE_TTL_SEC),
        cwd=cwd,
        repo_root=repo_root,
        git_dir=git_dir,
        head_mtime_ns=head_mtime_ns,
        index_mtime_ns=index_mtime_ns,
        dirty_cached=dirty_cached,
        context_static=context_static,
    )
    with _GIT_CONTEXT_CACHE_LOCK:
        global _git_context_cache
        _git_context_cache = entry


def _remaining_budget(deadline: float) -> float:
    """Return the seconds left before *deadline*, never negative."""
    return max(0.0, deadline - time.monotonic())


def _run_git_budgeted(
    args: list[str],
    *,
    repo_root: str | None,
    label: str,
    warnings: list[str],
    per_cmd_timeout_sec: float,
    deadline: float,
) -> str | None:
    """Run a git command within the remaining time budget.

    Failures, including an exhausted budget, are appended to *warnings* under
    *label*.

    Returns:
        The command's stdout, or None if it did not run or failed.
    """
    remaining = _remaining_budget(deadline)
    if remaining <= 0:
        warnings.append(f"{label}: budget exhausted")
        return None

    timeout = min(max(per_cmd_timeout_sec, 0.01), remaining)
    result = _run_git_command(args, timeout=timeout, repo_root=repo_root)
    if isinstance(result, str):
        return result
    warnings.append(f"{label}: {result.reason}")
    return None


def _extract_crash_frame(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Return the snapshot's crash frame, else its first frame, else None."""
    frames = snapshot.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    frames_list = cast(list[dict[str, Any]], frames)

    crash_idx_raw = snapshot.get("crash_frame_index", 0)
    try:
        crash_idx = int(crash_idx_raw)
    except (TypeError, ValueError):
        crash_idx = 0

    if 0 <= crash_idx < len(frames_list) and isinstance(frames_list[crash_idx], dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return frames_list[crash_idx]
    if isinstance(frames_list[0], dict):  # pyright: ignore[reportUnnecessaryIsInstance] — defense-in-depth for malformed data
        return frames_list[0]
    return None


def _extract_crash_line(crash_frame: dict[str, Any]) -> int | None:
    """Return the crash frame's line as a positive int, or None if unusable."""
    line_raw = crash_frame.get("line")
    if isinstance(line_raw, int):
        line = line_raw
    elif isinstance(line_raw, str):
        try:
            line = int(line_raw)
        except ValueError:
            return None
    else:
        return None
    return line if line > 0 else None


def _resolve_repo_relative_file(
    crash_frame: dict[str, Any],
    *,
    repo_root: str,
) -> tuple[str | None, bool]:
    """Resolve the crash frame's file relative to *repo_root*.

    Prefers the frame's ``file_rel`` and falls back to its absolute ``file``.

    Returns:
        ``(path, in_repo)``: the repo-relative POSIX path with True, or the raw
        path with False when it lies outside the repository; ``(None, False)``
        when the frame names no file.
    """
    repo_root_path = Path(repo_root).resolve()

    file_rel_raw = crash_frame.get("file_rel")
    if isinstance(file_rel_raw, str) and file_rel_raw:
        rel_candidate = file_rel_raw.replace("\\", "/")
        if not rel_candidate.startswith(".."):
            try:
                joined = (repo_root_path / rel_candidate).resolve()
                rel = joined.relative_to(repo_root_path).as_posix()
                return rel, True
            except (OSError, ValueError):
                pass
        return rel_candidate, False

    file_abs_raw = crash_frame.get("file")
    if isinstance(file_abs_raw, str) and file_abs_raw:
        try:
            file_abs = Path(file_abs_raw).resolve()
            rel = file_abs.relative_to(repo_root_path).as_posix()
            return rel, True
        except (OSError, ValueError):
            return file_abs_raw, False

    return None, False


def _parse_blame_porcelain(text: str, line: int) -> dict[str, Any] | None:
    """Parse ``git blame --porcelain`` output for one line into metadata.

    Returns:
        A dict with ``line``, ``commit``, and when present ``author``,
        ``author_time`` (UTC ISO 8601), and ``summary``; None if no commit was found.
    """
    if not text:
        return None

    result: dict[str, Any] = {"line": line}
    for raw_line in text.splitlines():
        line_text = raw_line.strip()
        if not line_text:
            continue

        if "commit" not in result and re.match(r"^[0-9a-f]{40,64}\s+\d+\s+\d+\s+\d+$", line_text):
            result["commit"] = line_text.split()[0]
            continue

        if line_text.startswith("author "):
            result["author"] = line_text.removeprefix("author ").strip()
            continue

        if line_text.startswith("author-time "):
            ts_raw = line_text.removeprefix("author-time ").strip()
            try:
                ts = int(ts_raw)
                result["author_time"] = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                )
            except ValueError:
                result["author_time"] = ts_raw
            continue

        if line_text.startswith("summary "):
            result["summary"] = line_text.removeprefix("summary ").strip()
            continue

    if "commit" not in result:
        return None
    return result


_SHORTSTAT_RE = re.compile(
    r"(?P<files>\d+)\s+files?\s+changed"
    r"(?:,\s+(?P<insertions>\d+)\s+insertions?\(\+\))?"
    r"(?:,\s+(?P<deletions>\d+)\s+deletions?\(-\))?"
)
_LOG_SEPARATOR = "__LLMDEBUG_COMMIT__"
_HEAD_INFO_FORMAT = "%h%n%H%n%D"


def _parse_shortstat(line: str) -> tuple[int, int, int]:
    """Parse a ``--shortstat`` line into ``(files_changed, insertions, deletions)``."""
    m = _SHORTSTAT_RE.search(line)
    if m is None:
        return 0, 0, 0

    files_changed = int(m.group("files") or 0)
    insertions = int(m.group("insertions") or 0)
    deletions = int(m.group("deletions") or 0)
    return files_changed, insertions, deletions


def _parse_recent_commits(log_text: str) -> list[dict[str, Any]]:
    """Parse separator-delimited ``git log --shortstat`` output into commit dicts."""
    if not log_text:
        return []

    lines = [line.rstrip() for line in log_text.splitlines()]
    commits: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != _LOG_SEPARATOR:
            i += 1
            continue

        if i + 5 >= len(lines):
            break

        short_commit = lines[i + 1].strip()
        full_commit = lines[i + 2].strip()
        author = lines[i + 3].strip()
        author_time = lines[i + 4].strip()
        summary = lines[i + 5].strip()

        i += 6
        files_changed = 0
        insertions = 0
        deletions = 0
        while i < len(lines) and lines[i].strip() != _LOG_SEPARATOR:
            line = lines[i].strip()
            if "file changed" in line or "files changed" in line:
                files_changed, insertions, deletions = _parse_shortstat(line)
                break
            i += 1

        commits.append(
            {
                "commit": short_commit,
                "commit_full": full_commit,
                "author": author,
                "author_time": author_time,
                "summary": summary,
                "files_changed": files_changed,
                "insertions": insertions,
                "deletions": deletions,
            }
        )

    return commits


def _parse_numstat(text: str) -> dict[str, Any]:
    """Sum ``--numstat`` output into file, insertion, and deletion counts plus ``touched``.

    Binary files count as changed but add no line counts.
    """
    files_changed = 0
    insertions = 0
    deletions = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files_changed += 1
        ins_raw, del_raw = parts[0], parts[1]
        if ins_raw.isdigit():
            insertions += int(ins_raw)
        if del_raw.isdigit():
            deletions += int(del_raw)

    return {
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
        "touched": files_changed > 0,
    }


def _parse_head_info_from_log(
    log_output: str,
) -> tuple[str | None, str | None, str | None, bool]:
    """Parse HEAD info from ``git log -1 --format=%h%n%H%n%D`` output.

    Returns:
        ``(commit_short, commit_full, branch, detached)``; *branch* is ``"HEAD"``
        and *detached* is True when HEAD points at no branch.
    """
    lines = log_output.split("\n", 2)
    commit_short = lines[0].strip() or None
    commit_full = (lines[1].strip() or None) if len(lines) > 1 else None

    branch: str | None = None
    detached = False
    if len(lines) > 2 and lines[2].strip():
        decorations = [part.strip() for part in lines[2].split(", ") if part.strip()]
        for part in decorations:
            if part.startswith("HEAD -> "):
                branch = part[len("HEAD -> ") :]
                break
        if branch is None and "HEAD" in decorations:
            # HEAD present without a symbolic ref target -> detached
            branch = "HEAD"
            detached = True
    return commit_short, commit_full, branch, detached


def _candidate_repo_dirs(snapshot: dict[str, Any], *, out_dir: str) -> list[str]:
    """Build an ordered candidate list to discover the snapshot's git repo."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(raw_path: str | Path | None, *, from_file: bool = False) -> None:
        """Append the resolved directory of *raw_path* unless it is empty or seen."""
        if raw_path is None:
            return
        raw_text = str(raw_path).strip()
        if not raw_text:
            return

        try:
            path = Path(raw_text).expanduser()
            if from_file:
                path = path.parent
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            return

        key = resolved.as_posix()
        if key in seen:
            return
        seen.add(key)
        candidates.append(key)

    crash_frame = _extract_crash_frame(snapshot)
    if crash_frame is not None:
        file_abs_raw = crash_frame.get("file")
        if isinstance(file_abs_raw, str) and file_abs_raw:
            file_path = Path(file_abs_raw)
            if file_path.is_absolute():
                _add(file_path, from_file=True)

    _add(out_dir)

    try:
        _add(Path(out_dir).expanduser().resolve().parent)
    except (OSError, RuntimeError, ValueError):
        pass

    env = snapshot.get("env")
    if isinstance(env, dict):
        env_dict = cast(dict[str, Any], env)
        cwd_raw = env_dict.get("cwd")
        if isinstance(cwd_raw, str) and cwd_raw:
            _add(cwd_raw)
    return candidates


def _discover_repo_root(
    snapshot: dict[str, Any],
    *,
    out_dir: str,
    warnings: list[str],
    per_cmd_timeout_sec: float,
    deadline: float,
) -> str | None:
    """Find the snapshot's repository root by trying each candidate directory in order.

    Returns:
        The first root ``git rev-parse --show-toplevel`` reports, or None (with a
        warning) when no candidate works or the budget runs out.
    """
    candidates = _candidate_repo_dirs(snapshot, out_dir=out_dir)
    if not candidates:
        warnings.append("repo root: no discovery candidates")
        return None

    failures: list[str] = []
    for candidate in candidates:
        remaining = _remaining_budget(deadline)
        if remaining <= 0:
            warnings.append("repo root: budget exhausted")
            return None

        timeout = min(max(per_cmd_timeout_sec, 0.01), remaining)
        result = _run_git_command(
            ["rev-parse", "--show-toplevel"],
            timeout=timeout,
            repo_root=candidate,
        )
        if isinstance(result, str):
            return result
        failures.append(f"{candidate} ({result.reason})")

    preview = "; ".join(failures[:2])
    if len(failures) > 2:
        preview = f"{preview}; +{len(failures) - 2} more"
    warnings.append(f"repo root: not found from candidates: {preview}")
    return None


def get_enhanced_git_context(
    snapshot: dict[str, Any],
    *,
    out_dir: str,
    max_commits: int = 5,
    per_cmd_timeout_sec: float = 0.4,
    total_budget_sec: float = 1.2,
) -> dict[str, Any] | None:
    """Compute enhanced git metadata on-demand for a specific snapshot.

    This is intentionally metadata-only and budgeted:
    - crash-line blame metadata (no code hunk)
    - recent commit metadata + shortstats
    - crash-file diffstat metadata (how the HEAD commit changed the crash file)

    The repository is discovered from the crash frame's file, *out_dir* and its
    parent, and the snapshot's recorded cwd, in that order.

    Args:
        snapshot: The snapshot dict whose crash frame and environment are used.
        out_dir: Snapshot output directory, also a repository discovery hint.
        max_commits: Number of recent commits to list (at least 1).
        per_cmd_timeout_sec: Timeout for each git command, capped by what remains of
            the total budget.
        total_budget_sec: Wall-clock budget for all git commands combined.

    Returns:
        A dict with ``repo_root`` and, where available, ``head`` (commit, branch,
        detached and dirty state), ``crash_file`` (repo-relative path, line, and
        blame), ``crash_file_diffstat``, ``recent_commits``, and ``_git_warnings``
        for steps that failed or ran out of budget. None if no repository was found.
    """
    warnings: list[str] = []
    deadline = time.monotonic() + max(total_budget_sec, 0.0)

    repo_root = _discover_repo_root(
        snapshot,
        out_dir=out_dir,
        warnings=warnings,
        per_cmd_timeout_sec=per_cmd_timeout_sec,
        deadline=deadline,
    )
    if repo_root is None:
        return None

    context: dict[str, Any] = {"repo_root": repo_root}
    head: dict[str, Any] = {}

    # Batch: commit short + full + branch in a single git log call.
    head_info = _run_git_budgeted(
        ["log", "-1", f"--format={_HEAD_INFO_FORMAT}"],
        repo_root=repo_root,
        label="head info",
        warnings=warnings,
        per_cmd_timeout_sec=per_cmd_timeout_sec,
        deadline=deadline,
    )
    if head_info is not None:
        h_short, h_full, h_branch, h_detached = _parse_head_info_from_log(head_info)
        if h_short:
            head["commit"] = h_short
        if h_full:
            head["commit_full"] = h_full
        if h_branch:
            head["branch"] = h_branch
            if h_detached:
                head["detached"] = True

    status = _run_git_budgeted(
        ["status", "--porcelain"],
        repo_root=repo_root,
        label="dirty status",
        warnings=warnings,
        per_cmd_timeout_sec=per_cmd_timeout_sec,
        deadline=deadline,
    )
    if status is not None:
        head["dirty"] = len(status) > 0

    if head:
        context["head"] = head

    crash_frame = _extract_crash_frame(snapshot)
    if crash_frame is not None:
        crash_line = _extract_crash_line(crash_frame)
        file_rel, in_repo = _resolve_repo_relative_file(crash_frame, repo_root=repo_root)
        crash_file: dict[str, Any] = {"file_rel": file_rel, "in_repo": in_repo}
        if crash_line is not None:
            crash_file["line"] = crash_line

        if file_rel and in_repo and crash_line is not None:
            blame_text = _run_git_budgeted(
                ["blame", "-L", f"{crash_line},{crash_line}", "--porcelain", "--", file_rel],
                repo_root=repo_root,
                label="blame",
                warnings=warnings,
                per_cmd_timeout_sec=per_cmd_timeout_sec,
                deadline=deadline,
            )
            if blame_text is not None:
                blame_meta = _parse_blame_porcelain(blame_text, crash_line)
                if blame_meta is not None:
                    crash_file["blame"] = blame_meta

            diffstat_text = _run_git_budgeted(
                ["show", "--numstat", "--format=", "--", file_rel],
                repo_root=repo_root,
                label="crash file diffstat",
                warnings=warnings,
                per_cmd_timeout_sec=per_cmd_timeout_sec,
                deadline=deadline,
            )
            if diffstat_text is not None:
                context["crash_file_diffstat"] = _parse_numstat(diffstat_text)

        context["crash_file"] = crash_file

    log_text = _run_git_budgeted(
        [
            "log",
            "-n",
            str(max(1, max_commits)),
            "--date=iso-strict",
            f"--pretty=format:{_LOG_SEPARATOR}%n%h%n%H%n%an%n%ad%n%s",
            "--shortstat",
        ],
        repo_root=repo_root,
        label="recent commits",
        warnings=warnings,
        per_cmd_timeout_sec=per_cmd_timeout_sec,
        deadline=deadline,
    )
    if log_text is not None:
        recent = _parse_recent_commits(log_text)
        if recent:
            context["recent_commits"] = recent

    if warnings:
        context["_git_warnings"] = warnings
    return context


def get_git_context(*, dirty_mode: GitDirtyMode = "always") -> dict[str, Any] | None:
    """Capture git repository context for the current working directory.

    Results are cached in-process for about a second; the cache is bypassed when
    the cwd leaves the repository or ``HEAD`` or the index changes.

    Args:
        dirty_mode: How to determine the ``dirty`` flag: ``"always"`` re-checks the
            working tree on every call, ``"cached"`` reuses the status from a cache
            hit, and ``"off"`` skips the check and omits the key.

    Returns:
        A dict with ``commit``, ``commit_full``, ``branch`` (``detached`` when HEAD
        is detached), ``dirty``, and ``_git_warnings`` if HEAD lookup failed; or
        None if not in a git repo or nothing could be read.

    Raises:
        ValueError: If *dirty_mode* is not ``"always"``, ``"cached"``, or ``"off"``.

    Example:
        A typical result::

            {
                "commit": "abc123",
                "commit_full": "abc123def456...",
                "branch": "main",
                "dirty": True
            }
    """
    if dirty_mode not in {"always", "cached", "off"}:
        raise ValueError("dirty_mode must be 'always', 'cached', or 'off'")

    cached_hit, cached_context = _load_cached_git_context(dirty_mode=dirty_mode)
    if cached_hit:
        return cached_context

    repo_root_result = _run_git_command(["rev-parse", "--show-toplevel"])
    if isinstance(repo_root_result, _GitError):
        # Not a git repo, or git not available
        _store_git_context_cache(None, repo_root=None)
        return None
    repo_root: str = repo_root_result  # type: ignore[assignment]

    context: dict[str, Any] = {}
    _warnings: list[str] = []

    # Batch: commit short + full + branch in a single git log call.
    head_info = _run_git_command(
        ["log", "-1", f"--format={_HEAD_INFO_FORMAT}"], repo_root=repo_root
    )
    if isinstance(head_info, str):
        h_short, h_full, h_branch, h_detached = _parse_head_info_from_log(head_info)
        if h_short:
            context["commit"] = h_short
        if h_full:
            context["commit_full"] = h_full
        if h_branch:
            context["branch"] = h_branch
            if h_detached:
                context["detached"] = True
    else:
        _warnings.append(f"head info: {head_info.reason}")

    if dirty_mode != "off":
        dirty = _detect_dirty_status(repo_root)
        if dirty is None:
            status = _run_git_command(["status", "--porcelain"], repo_root=repo_root)
            if isinstance(status, str):
                dirty = len(status) > 0
        if dirty is not None:
            context["dirty"] = dirty

    if _warnings:
        context["_git_warnings"] = _warnings

    result = context if context else None
    _store_git_context_cache(result, repo_root=repo_root)
    return result
