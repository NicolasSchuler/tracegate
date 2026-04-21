"""Helper-level branch coverage tests for git_context internals."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

import llmdebug.git_context as gc


@pytest.fixture(autouse=True)
def _reset_git_cache() -> None:
    gc._git_context_cache = None


def test_run_git_command_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gc.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["git"], returncode=2, stdout="", stderr=""
        ),
    )
    rc_fail = gc._run_git_command(["status"])
    assert isinstance(rc_fail, gc._GitError)
    assert "exit code 2" in rc_fail.reason

    monkeypatch.setattr(
        gc.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 1)),
    )
    timeout = gc._run_git_command(["status"])
    assert isinstance(timeout, gc._GitError)
    assert timeout.reason == "timeout"

    monkeypatch.setattr(
        gc.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError())
    )
    missing = gc._run_git_command(["status"])
    assert isinstance(missing, gc._GitError)
    assert missing.reason == "git not found"

    monkeypatch.setattr(
        gc.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(OSError("bad os"))
    )
    os_err = gc._run_git_command(["status"])
    assert isinstance(os_err, gc._GitError)
    assert "OSError:" in os_err.reason


def test_run_git_command_rc_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gc.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["git"], returncode=1, stdout="", stderr=""
        ),
    )
    assert gc._run_git_command_rc(["status"], repo_root="/repo") == 1

    monkeypatch.setattr(
        gc.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 1)),
    )
    assert isinstance(gc._run_git_command_rc(["status"]), gc._GitError)

    monkeypatch.setattr(
        gc.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError())
    )
    assert isinstance(gc._run_git_command_rc(["status"]), gc._GitError)

    monkeypatch.setattr(
        gc.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(OSError("bad os"))
    )
    assert isinstance(gc._run_git_command_rc(["status"]), gc._GitError)


def test_resolve_cwd_and_path_within_repo_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gc.Path, "cwd", classmethod(lambda _cls: (_ for _ in ()).throw(ValueError("bad cwd")))
    )
    assert gc._resolve_cwd() is None

    monkeypatch.setattr(
        gc.Path, "resolve", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad path"))
    )
    assert gc._path_within_repo("/tmp/a", "/tmp") is False


def test_resolve_git_dir_handles_directory_and_gitfile(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    git_dir = repo / ".git"
    git_dir.mkdir()
    assert gc._resolve_git_dir(str(repo)) == git_dir.resolve()

    git_dir.rmdir()
    git_dir.write_text("not-gitdir", encoding="utf-8")
    assert gc._resolve_git_dir(str(repo)) is None

    actual = repo / ".actual_git"
    actual.mkdir()
    git_dir.write_text("gitdir: .actual_git", encoding="utf-8")
    assert gc._resolve_git_dir(str(repo)) == actual.resolve()


def test_resolve_git_dir_handles_invalid_gitfile_and_resolution_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_file = repo / ".git"

    git_file.write_text("", encoding="utf-8")
    assert gc._resolve_git_dir(str(repo)) is None

    git_file.write_text("not-gitdir", encoding="utf-8")
    assert gc._resolve_git_dir(str(repo)) is None

    git_file.write_text("gitdir: ", encoding="utf-8")
    assert gc._resolve_git_dir(str(repo)) is None

    git_file.write_text("gitdir: .badgit", encoding="utf-8")
    (repo / ".badgit").mkdir()
    real_resolve = Path.resolve

    def _resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if str(self).endswith(".badgit"):
            raise OSError("bad resolve")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _resolve)
    assert gc._resolve_git_dir(str(repo)) is None


def test_git_signal_mtimes_and_untracked_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gc, "_resolve_git_dir", lambda _repo_root: None)
    assert gc._git_signal_mtimes("/repo") == (None, None, None)

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    monkeypatch.setattr(gc, "_resolve_git_dir", lambda _repo_root: git_dir)
    git_dir_s, head_ns, index_ns = gc._git_signal_mtimes("/repo")
    assert git_dir_s == str(git_dir)
    assert head_ns is None
    assert index_ns is None

    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: gc._GitError("bad"))
    assert gc._detect_untracked_status("/repo") is None
    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: 0)
    assert gc._detect_untracked_status("/repo") is True
    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: 1)
    assert gc._detect_untracked_status("/repo") is False
    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: 2)
    assert gc._detect_untracked_status("/repo") is None


def test_detect_dirty_status_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: gc._GitError("bad"))
    assert gc._detect_dirty_status("/repo") is None

    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: 1)
    assert gc._detect_dirty_status("/repo") is True

    monkeypatch.setattr(gc, "_run_git_command_rc", lambda *_a, **_k: 2)
    assert gc._detect_dirty_status("/repo") is None

    calls = {"count": 0}

    def _rc(*_a: Any, **_k: Any) -> int | gc._GitError:
        calls["count"] += 1
        if calls["count"] == 1:
            return 0
        if calls["count"] == 2:
            return 0
        return 1

    monkeypatch.setattr(gc, "_run_git_command_rc", _rc)
    monkeypatch.setattr(gc, "_detect_untracked_status", lambda _repo_root: False)
    assert gc._detect_dirty_status("/repo") is False


def test_detect_dirty_status_additional_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_rc(
        args: list[str], timeout: float = 2.0, repo_root: str | None = None
    ) -> int | gc._GitError:
        _ = args
        _ = timeout
        _ = repo_root
        calls["count"] += 1
        if calls["count"] == 1:
            return 0
        if calls["count"] == 2:
            return gc._GitError("bad staged")
        if calls["count"] == 3:
            return 0
        if calls["count"] == 4:
            return 1
        if calls["count"] == 5:
            return 0
        return 2

    monkeypatch.setattr(gc, "_run_git_command_rc", fake_rc)
    assert gc._detect_dirty_status("/repo") is None
    assert gc._detect_dirty_status("/repo") is True
    assert gc._detect_dirty_status("/repo") is None


def test_hydrate_context_with_dirty_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gc, "_detect_dirty_status", lambda _repo_root: None)
    monkeypatch.setattr(gc, "_run_git_command", lambda *_a, **_k: " M file.py")
    ctx = gc._hydrate_context_with_dirty({"commit": "abc"}, repo_root="/repo")
    assert ctx["dirty"] is True
    assert ctx["commit"] == "abc"

    monkeypatch.setattr(gc, "_run_git_command", lambda *_a, **_k: gc._GitError("bad"))
    ctx2 = gc._hydrate_context_with_dirty({"commit": "abc"}, repo_root="/repo")
    assert "dirty" not in ctx2


def test_load_cached_git_context_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.monotonic()
    gc._git_context_cache = gc._GitContextCacheEntry(
        expires_at=now - 1,
        cwd="/cwd",
        repo_root=None,
        git_dir=None,
        head_mtime_ns=None,
        index_mtime_ns=None,
        dirty_cached=None,
        context_static=None,
    )
    assert gc._load_cached_git_context(dirty_mode="always") == (False, None)

    gc._git_context_cache = gc._GitContextCacheEntry(
        expires_at=time.monotonic() + 10,
        cwd="/expected",
        repo_root=None,
        git_dir=None,
        head_mtime_ns=None,
        index_mtime_ns=None,
        dirty_cached=None,
        context_static=None,
    )
    monkeypatch.setattr(gc, "_resolve_cwd", lambda: "/other")
    assert gc._load_cached_git_context(dirty_mode="always") == (False, None)

    monkeypatch.setattr(gc, "_resolve_cwd", lambda: "/expected")
    assert gc._load_cached_git_context(dirty_mode="always") == (True, None)

    gc._git_context_cache = gc._GitContextCacheEntry(
        expires_at=time.monotonic() + 10,
        cwd="/cwd",
        repo_root="/repo",
        git_dir="/repo/.git",
        head_mtime_ns=1,
        index_mtime_ns=2,
        dirty_cached=True,
        context_static={"commit": "abc"},
    )
    monkeypatch.setattr(gc, "_resolve_cwd", lambda: "/repo/subdir")
    monkeypatch.setattr(gc, "_path_within_repo", lambda *_a, **_k: True)
    monkeypatch.setattr(gc, "_git_signal_mtimes", lambda _repo_root: ("/repo/.git", 1, 2))

    hit, context_off = gc._load_cached_git_context(dirty_mode="off")
    assert hit is True
    assert context_off == {"commit": "abc"}

    hit_cached, context_cached = gc._load_cached_git_context(dirty_mode="cached")
    assert hit_cached is True
    assert context_cached == {"commit": "abc", "dirty": True}

    monkeypatch.setattr(
        gc,
        "_hydrate_context_with_dirty",
        lambda context_static, *, repo_root: {**context_static, "dirty": False},
    )
    hit_always, context_always = gc._load_cached_git_context(dirty_mode="always")
    assert hit_always is True
    assert context_always == {"commit": "abc", "dirty": False}


def test_load_cached_git_context_additional_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    gc._git_context_cache = gc._GitContextCacheEntry(
        expires_at=time.monotonic() + 10,
        cwd="/repo",
        repo_root="/repo",
        git_dir="/repo/.git",
        head_mtime_ns=1,
        index_mtime_ns=2,
        dirty_cached=None,
        context_static=None,
    )
    monkeypatch.setattr(gc, "_resolve_cwd", lambda: "/outside")
    assert gc._load_cached_git_context(dirty_mode="always") == (False, None)

    gc._git_context_cache = gc._GitContextCacheEntry(
        expires_at=time.monotonic() + 10,
        cwd="/repo",
        repo_root="/repo",
        git_dir="/repo/.git",
        head_mtime_ns=1,
        index_mtime_ns=2,
        dirty_cached=None,
        context_static=None,
    )
    monkeypatch.setattr(gc, "_resolve_cwd", lambda: "/repo/subdir")
    monkeypatch.setattr(gc, "_path_within_repo", lambda *_a, **_k: True)
    monkeypatch.setattr(gc, "_git_signal_mtimes", lambda _repo_root: ("/repo/.git", 1, 2))
    assert gc._load_cached_git_context(dirty_mode="always") == (True, None)


def test_store_git_context_cache_records_static_and_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gc, "_resolve_cwd", lambda: "/cwd")
    monkeypatch.setattr(gc, "_git_signal_mtimes", lambda _repo_root: ("/repo/.git", 11, 22))
    monkeypatch.setattr(gc.time, "monotonic", lambda: 100.0)

    gc._store_git_context_cache({"commit": "abc", "dirty": True}, repo_root="/repo")
    assert gc._git_context_cache is not None
    assert gc._git_context_cache.repo_root == "/repo"
    assert gc._git_context_cache.context_static == {"commit": "abc"}
    assert gc._git_context_cache.dirty_cached is True

    gc._store_git_context_cache(None, repo_root=None)
    assert gc._git_context_cache is not None
    assert gc._git_context_cache.repo_root is None


def test_run_git_budgeted_handles_budget_and_command_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    exhausted = gc._run_git_budgeted(
        ["status"],
        repo_root=None,
        label="status",
        warnings=warnings,
        per_cmd_timeout_sec=0.4,
        deadline=time.monotonic() - 1.0,
    )
    assert exhausted is None
    assert warnings == ["status: budget exhausted"]

    warnings2: list[str] = []
    monkeypatch.setattr(gc, "_run_git_command", lambda *_a, **_k: gc._GitError("boom"))
    failed = gc._run_git_budgeted(
        ["status"],
        repo_root=None,
        label="status",
        warnings=warnings2,
        per_cmd_timeout_sec=0.4,
        deadline=time.monotonic() + 1.0,
    )
    assert failed is None
    assert warnings2 == ["status: boom"]


def test_extract_crash_frame_line_and_repo_file_resolution(tmp_path: Path) -> None:
    assert gc._extract_crash_frame({"frames": "bad"}) is None
    assert gc._extract_crash_frame({"frames": []}) is None
    assert gc._extract_crash_frame({"frames": ["bad"]}) is None

    snapshot = {"frames": [{"line": "12"}, {"line": 9}], "crash_frame_index": "bad"}
    assert gc._extract_crash_frame(snapshot) == {"line": "12"}
    assert gc._extract_crash_line({"line": 5}) == 5
    assert gc._extract_crash_line({"line": "7"}) == 7
    assert gc._extract_crash_line({"line": "bad"}) is None
    assert gc._extract_crash_line({"line": -1}) is None

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    file_inside = repo_root / "a.py"
    file_inside.write_text("x=1", encoding="utf-8")
    rel, in_repo = gc._resolve_repo_relative_file({"file_rel": "a.py"}, repo_root=str(repo_root))
    assert (rel, in_repo) == ("a.py", True)
    rel2, in_repo2 = gc._resolve_repo_relative_file(
        {"file_rel": "../outside.py"}, repo_root=str(repo_root)
    )
    assert rel2 == "../outside.py"
    assert in_repo2 is False
    rel3, in_repo3 = gc._resolve_repo_relative_file(
        {"file": str(file_inside)}, repo_root=str(repo_root)
    )
    assert (rel3, in_repo3) == ("a.py", True)
    rel4, in_repo4 = gc._resolve_repo_relative_file(
        {"file": "/tmp/outside.py"}, repo_root=str(repo_root)
    )
    assert rel4 == "/tmp/outside.py"
    assert in_repo4 is False
    assert gc._resolve_repo_relative_file({}, repo_root=str(repo_root)) == (None, False)


def test_extract_crash_frame_and_relative_resolution_fallbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = {
        "frames": [{"function": "fallback"}, "bad"],
        "crash_frame_index": 1,
    }
    assert gc._extract_crash_frame(snapshot) == {"function": "fallback"}

    real_relative_to = Path.relative_to

    def _relative_to(self: Path, *args: Any, **kwargs: Any) -> Path:
        if str(self).endswith("broken.py"):
            raise ValueError("bad relative")
        return real_relative_to(self, *args, **kwargs)

    monkeypatch.setattr(Path, "relative_to", _relative_to)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    rel, in_repo = gc._resolve_repo_relative_file(
        {"file_rel": "broken.py"}, repo_root=str(repo_root)
    )
    assert rel == "broken.py"
    assert in_repo is False


def test_parse_blame_shortstat_recent_and_numstat_helpers() -> None:
    assert gc._parse_blame_porcelain("", 10) is None
    assert gc._parse_blame_porcelain("author Alice", 10) is None

    blame = gc._parse_blame_porcelain(
        "f" * 40 + " 12 12 1\nauthor Alice\nauthor-time not-int\nsummary Fix line\n",
        12,
    )
    assert blame is not None
    assert blame["commit"] == "f" * 40
    assert blame["author_time"] == "not-int"

    blame_sha256 = gc._parse_blame_porcelain(
        "a" * 64 + " 8 8 1\nauthor Bob\nsummary sha256 repo\n",
        8,
    )
    assert blame_sha256 is not None
    assert blame_sha256["commit"] == "a" * 64

    assert gc._parse_shortstat("nonsense") == (0, 0, 0)
    assert gc._parse_shortstat("2 files changed, 3 insertions(+), 1 deletion(-)") == (2, 3, 1)

    assert gc._parse_recent_commits("") == []
    assert gc._parse_recent_commits("__LLMDEBUG_COMMIT__\nonly\n") == []
    recent = gc._parse_recent_commits(
        "__LLMDEBUG_COMMIT__\nabc1234\n" + "a" * 40 + "\nAlice\n2026-01-01T00:00:00Z\nFix\n"
        " 1 file changed, 2 insertions(+), 0 deletions(-)\n"
    )
    assert len(recent) == 1
    assert recent[0]["commit"] == "abc1234"
    assert recent[0]["files_changed"] == 1

    numstat = gc._parse_numstat("\n10\t2\ta.py\n-\t-\tbinary.dat\nbad-line\n")
    assert numstat["files_changed"] == 2
    assert numstat["insertions"] == 10
    assert numstat["deletions"] == 2
    assert numstat["touched"] is True


def test_candidate_repo_dirs_and_discover_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    crash_file = tmp_path / "repo" / "src" / "app.py"
    crash_file.parent.mkdir(parents=True)
    crash_file.write_text("x=1", encoding="utf-8")
    snapshot = {
        "crash_frame_index": 0,
        "frames": [{"file": str(crash_file)}],
        "env": {"cwd": str(tmp_path / "repo")},
    }
    dirs = gc._candidate_repo_dirs(snapshot, out_dir=str(tmp_path / "repo" / ".llmdebug"))
    assert dirs
    assert len(dirs) == len(set(dirs))
    assert str(Path.cwd().resolve()) not in dirs

    warnings: list[str] = []
    monkeypatch.setattr(gc, "_candidate_repo_dirs", lambda *_a, **_k: [])
    assert (
        gc._discover_repo_root(
            snapshot,
            out_dir=".llmdebug",
            warnings=warnings,
            per_cmd_timeout_sec=0.1,
            deadline=time.monotonic() + 1.0,
        )
        is None
    )
    assert "no discovery candidates" in warnings[0]

    warnings2: list[str] = []
    monkeypatch.setattr(gc, "_candidate_repo_dirs", lambda *_a, **_k: ["/a", "/b", "/c"])
    monkeypatch.setattr(gc, "_remaining_budget", lambda _deadline: 1.0)
    monkeypatch.setattr(gc, "_run_git_command", lambda *_a, **_k: gc._GitError("not repo"))
    assert (
        gc._discover_repo_root(
            snapshot,
            out_dir=".llmdebug",
            warnings=warnings2,
            per_cmd_timeout_sec=0.1,
            deadline=time.monotonic() + 1.0,
        )
        is None
    )
    assert "not found from candidates" in warnings2[-1]


def test_get_enhanced_git_context_attaches_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 12}],
    }

    def _discover_repo_root(
        _snapshot: dict[str, Any],
        *,
        out_dir: str,
        warnings: list[str],
        per_cmd_timeout_sec: float,
        deadline: float,
    ) -> str | None:
        _ = out_dir, per_cmd_timeout_sec, deadline
        warnings.append("repo root: not found from candidates: /repo (boom)")
        return "/repo"

    monkeypatch.setattr(gc, "_discover_repo_root", _discover_repo_root)
    monkeypatch.setattr(gc, "_run_git_budgeted", lambda *_a, **_k: None)
    context = gc.get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    assert context["_git_warnings"] == ["repo root: not found from candidates: /repo (boom)"]


def test_get_enhanced_and_basic_git_context_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": "12"}],
    }

    monkeypatch.setattr(gc, "_discover_repo_root", lambda *_a, **_k: "/repo")

    values = {
        "head info": f"{'f' * 7}\n{'f' * 40}\nHEAD",
        "dirty status": "",
        "blame": "not-porcelain",
        "crash file diffstat": "10\t2\tapp.py\n",
        "recent commits": "",
    }

    def _budgeted(
        _args: list[str],
        *,
        repo_root: str | None,
        label: str,
        warnings: list[str],
        per_cmd_timeout_sec: float,
        deadline: float,
    ) -> str | None:
        del repo_root, warnings, per_cmd_timeout_sec, deadline
        return values[label]

    monkeypatch.setattr(gc, "_run_git_budgeted", _budgeted)
    context = gc.get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    assert context["head"]["commit"] == "f" * 7
    assert context["head"]["commit_full"] == "f" * 40
    assert context["head"]["branch"] == "HEAD"
    assert context["head"]["detached"] is True
    assert context["head"]["dirty"] is False
    assert "blame" not in context["crash_file"]
    assert context["crash_file_diffstat"]["files_changed"] == 1

    monkeypatch.setattr(gc, "_load_cached_git_context", lambda *, dirty_mode: (False, None))

    def _run_for_basic(
        args: list[str], timeout: float = 2.0, repo_root: str | None = None
    ) -> str | gc._GitError:
        del timeout, repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[0] == "log" and args[1] == "-1":
            return gc._GitError("no commits")
        if args[:2] == ["status", "--porcelain"]:
            return ""
        return gc._GitError("unexpected")

    monkeypatch.setattr(gc, "_run_git_command", _run_for_basic)
    monkeypatch.setattr(gc, "_detect_dirty_status", lambda _repo_root: None)
    monkeypatch.setattr(gc, "_store_git_context_cache", lambda *_a, **_k: None)
    basic = gc.get_git_context(dirty_mode="always")
    assert basic is not None
    assert "commit" not in basic
    assert "branch" not in basic
    assert basic["dirty"] is False
    assert basic["_git_warnings"] == ["head info: no commits"]
