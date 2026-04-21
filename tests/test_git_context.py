"""Tests for git context capture."""

import os
import subprocess
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

import llmdebug.git_context as git_context_module
from llmdebug.git_context import _GitError, get_enhanced_git_context, get_git_context


@pytest.fixture(autouse=True)
def _clear_git_context_cache():
    """Prevent cross-test leakage from git context cache."""
    git_context_module._git_context_cache = None
    yield
    git_context_module._git_context_cache = None


def test_git_context_in_repo():
    """Test git context capture when in a git repository."""
    # This test runs from the llmdebug repo itself
    context = get_git_context()

    # Should have context since we're in a git repo
    assert context is not None
    assert "commit" in context
    assert "commit_full" in context
    assert "branch" in context
    assert "dirty" in context

    # Commit should be a short hash (7+ chars)
    assert len(context["commit"]) >= 7
    # Full commit should be 40 chars
    assert len(context["commit_full"]) == 40


def test_git_context_not_in_repo(tmp_path, monkeypatch):
    """Test git context capture when not in a git repository."""
    # Change to a non-git directory
    monkeypatch.chdir(tmp_path)

    context = get_git_context()

    # Should return None when not in a git repo
    assert context is None


def test_git_context_dirty_status(tmp_path):
    """Test that dirty status is correctly detected."""
    # Initialize a new git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Create and commit a file
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Save current dir and change to temp repo
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Should be clean
        context = get_git_context()
        assert context is not None
        assert context["dirty"] is False

        # Untracked files should count as dirty.
        untracked_file = tmp_path / "untracked.txt"
        untracked_file.write_text("new")
        context = get_git_context()
        assert context is not None
        assert context["dirty"] is True

        # Return to clean before testing tracked-file edits.
        untracked_file.unlink()
        context = get_git_context()
        assert context is not None
        assert context["dirty"] is False

        # Tracked-file edits should also count as dirty.
        test_file.write_text("modified")
        context = get_git_context()
        assert context is not None
        assert context["dirty"] is True
    finally:
        os.chdir(old_cwd)


def test_git_context_detached_head(tmp_path):
    """Test git context when HEAD is detached."""
    # Initialize a new git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Create and commit a file
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Detach HEAD
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_hash = result.stdout.strip()
    subprocess.run(
        ["git", "checkout", commit_hash],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Save current dir and change to temp repo
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        context = get_git_context()
        assert context is not None
        assert context.get("detached") is True
    finally:
        os.chdir(old_cwd)


def test_parse_head_info_from_log_preserves_commas_in_branch_name() -> None:
    commit_short, commit_full, branch, detached = git_context_module._parse_head_info_from_log(
        "abc1234\n" + ("a" * 40) + "\nHEAD -> feat,comma, origin/feat,comma"
    )
    assert commit_short == "abc1234"
    assert commit_full == "a" * 40
    assert branch == "feat,comma"
    assert detached is False


def test_enhanced_git_context_parses_metadata(monkeypatch):
    snapshot = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 42}],
    }

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[:2] == ["log", "-1"]:
            return "abc1234\nabc1234567890abcdef1234567890abcdef1234\nHEAD -> main"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "blame":
            return (
                "def56789abcdef0123456789abcdef0123456789 42 42 1\n"
                "author Alice\n"
                "author-time 1700000000\n"
                "summary Fix parser line\n"
                "filename app.py\n"
                "\treturn int(value)\n"
            )
        if args and args[0] == "log":
            return (
                "__LLMDEBUG_COMMIT__\n"
                "abc1234\n"
                "abc1234567890abcdef1234567890abcdef1234\n"
                "Alice\n"
                "2026-02-10T10:00:00+00:00\n"
                "Fix parser\n"
                " 1 file changed, 2 insertions(+), 1 deletion(-)\n"
            )
        if args and args[0] == "show":
            return "12\t3\tapp.py\n"
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(
        snapshot,
        out_dir=".llmdebug",
        max_commits=5,
        per_cmd_timeout_sec=1.0,
        total_budget_sec=5.0,
    )
    assert context is not None
    assert context["head"]["commit"] == "abc1234"
    assert context["crash_file"]["file_rel"] == "app.py"
    assert context["crash_file"]["blame"]["author"] == "Alice"
    assert len(context["recent_commits"]) == 1
    assert context["crash_file_diffstat"]["files_changed"] == 1


def test_enhanced_git_context_discovers_repo_from_out_dir(monkeypatch, tmp_path):
    target_repo = (tmp_path / "target_repo").resolve()
    target_repo_str = target_repo.as_posix()
    target_out_dir = (target_repo / ".llmdebug").resolve()
    target_out_dir_str = target_out_dir.as_posix()
    wrong_repo_str = (tmp_path / "wrong_repo").resolve().as_posix()
    snapshot = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py"}],
    }

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        repo = str(repo_root) if repo_root is not None else None
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            if repo in {target_out_dir_str, target_repo_str}:
                return target_repo_str
            return wrong_repo_str
        if args[:2] == ["log", "-1"]:
            if repo == target_repo_str:
                return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
            return "badcafe\n" + ("b" * 40) + "\nHEAD -> wrong-branch"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "log":
            return ""
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=target_out_dir_str)
    assert context is not None
    assert context["repo_root"] == target_repo_str
    assert context["head"]["commit"] == "abc1234"
    assert context["head"]["branch"] == "main"


def test_enhanced_git_context_marks_outside_repo(monkeypatch):
    snapshot = {
        "crash_frame_index": 0,
        "frames": [{"file": "/tmp/outside.py", "line": 8}],
    }
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        calls.append((args, repo_root))
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[:2] == ["log", "-1"]:
            return "abc1234\nabc1234567890abcdef1234567890abcdef1234\nHEAD -> main"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "log":
            return ""
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    assert context["crash_file"]["in_repo"] is False
    assert "blame" not in context["crash_file"]
    assert all(args[0] not in {"blame", "show"} for args, _ in calls)


def test_enhanced_git_context_budget_exhausted(monkeypatch):
    snapshot = {"frames": [{"file_rel": "app.py", "line": 1}], "crash_frame_index": 0}

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = args
        _ = timeout
        _ = repo_root
        return "/repo"

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=".llmdebug", total_budget_sec=0.0)
    assert context is None


def test_git_context_cache_reuses_static_data(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(git_context_module, "_GIT_CONTEXT_CACHE_TTL_SEC", 60.0)

    run_calls: list[tuple[list[str], str | None]] = []
    rc_calls: list[tuple[list[str], str | None]] = []

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        run_calls.append((args, repo_root))
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        rc_calls.append((args, repo_root))
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    first = get_git_context()
    second = get_git_context()

    assert first is not None
    assert second is not None
    assert first["commit"] == "abc1234"
    assert second["commit"] == "abc1234"

    # Static metadata should be fetched once (cache hit on second call).
    static_calls = [args for args, _ in run_calls if args[:2] == ["rev-parse", "--show-toplevel"]]
    assert len(static_calls) == 1

    # Dirty checks still run on each access to avoid stale status.
    diff_calls = [args for args, _ in rc_calls if args[:1] == ["diff-files"]]
    assert len(diff_calls) == 2


def test_git_context_cache_invalidates_on_head_change(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    head_path = git_dir / "HEAD"
    head_path.write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(git_context_module, "_GIT_CONTEXT_CACHE_TTL_SEC", 60.0)

    commit_state = {"short": "aaa1111", "full": "a" * 40}
    show_toplevel_calls = 0

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        nonlocal show_toplevel_calls
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            show_toplevel_calls += 1
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return f"{commit_state['short']}\n{commit_state['full']}\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = args
        _ = timeout
        _ = repo_root
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    first = get_git_context()
    assert first is not None
    assert first["commit"] == "aaa1111"
    assert show_toplevel_calls == 1

    commit_state["short"] = "bbb2222"
    commit_state["full"] = "b" * 40
    # Head mtime change should invalidate cache and trigger refresh.
    head_path.write_text("ref: refs/heads/main\n# changed\n", encoding="utf-8")
    second = get_git_context()
    assert second is not None
    assert second["commit"] == "bbb2222"
    assert show_toplevel_calls == 2


def test_git_context_dirty_mode_off_skips_dirty_detection(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(git_context_module, "_GIT_CONTEXT_CACHE_TTL_SEC", 60.0)

    run_calls: list[list[str]] = []
    rc_calls: list[list[str]] = []

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        run_calls.append(args)
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        rc_calls.append(args)
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    context = get_git_context(dirty_mode="off")
    assert context is not None
    assert "dirty" not in context
    assert rc_calls == []
    assert all(call[:2] != ["status", "--porcelain"] for call in run_calls)


def test_git_context_dirty_mode_cached_reuses_dirty(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(git_context_module, "_GIT_CONTEXT_CACHE_TTL_SEC", 60.0)

    rc_calls: list[list[str]] = []

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        rc_calls.append(args)
        # Mark repo dirty on first detection path.
        return 1

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    first = get_git_context(dirty_mode="cached")
    second = get_git_context(dirty_mode="cached")
    assert first is not None
    assert second is not None
    assert first.get("dirty") is True
    assert second.get("dirty") is True

    diff_files_calls = [call for call in rc_calls if call[:1] == ["diff-files"]]
    assert len(diff_files_calls) == 1


def test_git_context_dirty_mode_cached_after_off_recomputes_dirty(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(git_context_module, "_GIT_CONTEXT_CACHE_TTL_SEC", 60.0)

    run_calls: list[list[str]] = []
    rc_calls: list[list[str]] = []

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        run_calls.append(args)
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        rc_calls.append(args)
        if args[:1] == ["diff-files"]:
            return 0
        if args[:1] == ["diff-index"]:
            return 0
        if args[:1] == ["ls-files"]:
            return 1
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    off_context = get_git_context(dirty_mode="off")
    cached_context = get_git_context(dirty_mode="cached")

    assert off_context is not None
    assert "dirty" not in off_context
    assert cached_context is not None
    assert cached_context.get("dirty") is False

    show_toplevel_calls = [
        call for call in run_calls if call[:2] == ["rev-parse", "--show-toplevel"]
    ]
    assert len(show_toplevel_calls) == 2
    ls_files_calls = [call for call in rc_calls if call[:1] == ["ls-files"]]
    assert len(ls_files_calls) == 1


def test_git_context_rejects_invalid_dirty_mode():
    with pytest.raises(ValueError, match="dirty_mode must be"):
        get_git_context(dirty_mode=cast(Any, "invalid"))


# ---------- Coverage-gap tests below ----------


def test_resolve_git_dir_not_a_dir_not_a_file(tmp_path):
    """Cover L126-127: .git path is neither dir nor file → None."""
    # No .git entry at all in tmp_path
    result = git_context_module._resolve_git_dir(str(tmp_path))
    assert result is None


def test_resolve_git_dir_gitdir_absolute_path(tmp_path):
    """Cover L140-143: .git file with an absolute gitdir path."""
    actual_git_dir = tmp_path / "real_git_dir"
    actual_git_dir.mkdir()
    git_file = tmp_path / ".git"
    git_file.write_text(f"gitdir: {actual_git_dir}\n", encoding="utf-8")
    result = git_context_module._resolve_git_dir(str(tmp_path))
    assert result == actual_git_dir.resolve()


def test_parse_blame_porcelain_with_empty_lines():
    """Cover L402-403: blank lines in blame output are skipped."""
    text = "\ndef56789abcdef0123456789abcdef0123456789 10 10 1\n\nauthor Alice\n\nsummary Fix bug\n"
    result = git_context_module._parse_blame_porcelain(text, 10)
    assert result is not None
    assert result["author"] == "Alice"
    assert result["commit"] == "def56789abcdef0123456789abcdef0123456789"


def test_parse_recent_commits_lines_before_shortstat():
    """Cover L478-485, L480-483: inner loop iterates past non-shortstat lines."""
    sep = git_context_module._LOG_SEPARATOR
    log_text = (
        f"{sep}\n"
        "abc1234\n"
        "abc1234567890abcdef1234567890abcdef1234\n"
        "Alice\n"
        "2026-02-10T10:00:00+00:00\n"
        "Fix parser\n"
        "some-extra-noise\n"
        "another-noise\n"
        " 1 file changed, 2 insertions(+), 1 deletion(-)\n"
        f"{sep}\n"
    )
    commits = git_context_module._parse_recent_commits(log_text)
    assert len(commits) == 1
    assert commits[0]["files_changed"] == 1


def test_parse_recent_commits_no_shortstat_line():
    """Cover branch [478,485]: while loop exits without finding shortstat."""
    sep = git_context_module._LOG_SEPARATOR
    log_text = (
        f"{sep}\n"
        "abc1234\n"
        "abc1234567890abcdef1234567890abcdef1234\n"
        "Alice\n"
        "2026-02-10T10:00:00+00:00\n"
        "Fix parser\n"
        f"{sep}\n"
    )
    commits = git_context_module._parse_recent_commits(log_text)
    assert len(commits) == 1
    assert commits[0]["files_changed"] == 0


def test_parse_head_info_no_decorations():
    """Cover branch [542,552]: only 2 lines or empty third line → no branch."""
    short, full, branch, detached = git_context_module._parse_head_info_from_log(
        "abc1234\n" + ("a" * 40)
    )
    assert short == "abc1234"
    assert full == "a" * 40
    assert branch is None
    assert detached is False


def test_parse_head_info_empty_third_line():
    """Cover branch [542,552]: third line is whitespace → no branch."""
    _short, _full, branch, detached = git_context_module._parse_head_info_from_log(
        "abc1234\n" + ("a" * 40) + "\n   "
    )
    assert branch is None
    assert detached is False


def test_candidate_repo_dirs_no_frames(monkeypatch):
    """Cover branches [582,589] and [708,742]: snapshot with no frames."""
    snapshot: dict[str, Any] = {"frames": []}
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    # Should still have out_dir-derived candidates, but no crash-file candidate
    assert isinstance(candidates, list)


def test_candidate_repo_dirs_relative_file():
    """Cover branch [586,589]: crash frame has a relative (non-absolute) file path."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file": "relative/path.py", "line": 1}],
    }
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    # Relative file path should be skipped; only out_dir candidates present
    assert isinstance(candidates, list)
    assert not any("relative/path.py" in c for c in candidates)


def test_candidate_repo_dirs_none_file():
    """Cover branches [561,562]: _add called with None raw_path."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"line": 1}],  # no file or file_rel key
    }
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    assert isinstance(candidates, list)


def test_candidate_repo_dirs_empty_file():
    """Cover branches [564,565]: _add called with empty string."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file": "", "line": 1}],
    }
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    assert isinstance(candidates, list)


def test_candidate_repo_dirs_env_cwd(monkeypatch):
    """Cover branch [600,602]: snapshot with env.cwd adds candidate."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
        "env": {"cwd": "/custom/work/dir"},
    }
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    assert any("custom/work/dir" in c for c in candidates)


def test_candidate_repo_dirs_path_resolve_error(monkeypatch):
    """Cover L572-573: OSError/ValueError during path resolution in _add."""
    original_expanduser = Path.expanduser

    def _bad_expanduser(self: Path) -> Path:
        if "BADPATH" in str(self):
            raise OSError("synthetic path error")
        return original_expanduser(self)

    monkeypatch.setattr(Path, "expanduser", _bad_expanduser)
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file": "/BADPATH/crash.py", "line": 1}],
    }
    # Should not raise, the error is caught internally
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    assert isinstance(candidates, list)


def test_candidate_repo_dirs_parent_resolve_error(monkeypatch):
    """Cover L593-594: OSError resolving parent of out_dir."""
    original_resolve = Path.resolve

    call_count = {"n": 0}

    def _bad_resolve(self: Path, strict: bool = False) -> Path:
        if "BADOUTDIR" in str(self):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise OSError("synthetic resolve error")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _bad_resolve)
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
    }
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/BADOUTDIR/sub")
    assert isinstance(candidates, list)


def test_store_cache_with_none_context_and_valid_repo(monkeypatch, tmp_path):
    """Cover branch [284,290]: context is None but repo_root is set."""
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    git_context_module._store_git_context_cache(None, repo_root=str(repo_root))
    entry = git_context_module._git_context_cache
    assert entry is not None
    assert entry.context_static is None
    assert entry.repo_root == str(repo_root)


def test_enhanced_git_context_partial_head_info(monkeypatch):
    """Cover branches [684-693]: head_info with missing short/full/branch."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
    }

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[:2] == ["log", "-1"]:
            # Return empty commit short, empty full, empty decorations
            return "\n\n"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "log":
            return ""
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    # head should have no commit/commit_full/branch keys since they were empty
    assert "commit" not in context.get("head", {})
    assert "commit_full" not in context.get("head", {})
    assert "branch" not in context.get("head", {})


def test_enhanced_git_context_no_crash_frame(monkeypatch):
    """Cover branch [708,742]: snapshot with empty frames → no crash_file section."""
    snapshot: dict[str, Any] = {"frames": [], "crash_frame_index": 0}

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "log":
            return ""
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    assert "crash_file" not in context


def test_get_git_context_partial_head_info(monkeypatch, tmp_path):
    """Cover branches [804-815]: get_git_context with empty short/full/branch."""
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            # Empty short, empty full, empty decorations
            return "\n\n"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    context = get_git_context()
    # Context should exist but have no commit/commit_full/branch
    assert context is not None
    assert "commit" not in context
    assert "commit_full" not in context
    assert "branch" not in context


def test_get_git_context_dirty_from_detect(monkeypatch, tmp_path):
    """Cover branch [819,821]: _detect_dirty_status returns non-None directly."""
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        # diff-files returns 1 → dirty=True, no need for porcelain fallback
        if args[:1] == ["diff-files"]:
            return 1
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    context = get_git_context()
    assert context is not None
    assert context["dirty"] is True


def test_get_git_context_dirty_both_none(monkeypatch, tmp_path):
    """Cover branch [821,824]: both dirty detection and porcelain return None."""
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        if args[:2] == ["status", "--porcelain"]:
            # Porcelain also fails
            return _GitError("porcelain failed")
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        # _detect_dirty_status returns None (all commands return _GitError)
        return _GitError("error")

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    context = get_git_context()
    assert context is not None
    # dirty should not be set at all since both detections failed
    assert "dirty" not in context


def test_candidate_repo_dirs_empty_out_dir():
    """Cover L564-565: _add called with whitespace-only out_dir → early return."""
    snapshot: dict[str, Any] = {"frames": [], "crash_frame_index": 0}
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="   ")
    # out_dir "   " strips to "" → _add returns early, but parent resolution may still work
    assert isinstance(candidates, list)


def test_candidate_repo_dirs_env_cwd_not_string():
    """Cover branch [600,602] false: env.cwd is not a string → skip _add."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
        "env": {"cwd": 12345},
    }
    candidates = git_context_module._candidate_repo_dirs(snapshot, out_dir="/some/dir")
    assert isinstance(candidates, list)


def test_enhanced_git_context_head_info_none(monkeypatch):
    """Cover branch [682,693]: head_info is None → skip head parsing entirely."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
    }

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[:2] == ["log", "-1"]:
            return _GitError("log failed")
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "log":
            return ""
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    # head should be empty or missing commit keys
    head = context.get("head", {})
    assert "commit" not in head


def test_enhanced_git_context_detached_head(monkeypatch):
    """Cover L691: h_detached=True → head['detached'] = True in enhanced context."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
    }

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return "/repo"
        if args[:2] == ["log", "-1"]:
            # HEAD without a symbolic ref → detached
            return "abc1234\n" + ("a" * 40) + "\nHEAD"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args and args[0] == "log":
            return ""
        return _GitError("unexpected")

    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    context = get_enhanced_git_context(snapshot, out_dir=".llmdebug")
    assert context is not None
    assert context["head"]["detached"] is True
    assert context["head"]["branch"] == "HEAD"


def test_get_git_context_head_info_is_git_error(monkeypatch, tmp_path):
    """Cover L813: head_info returns _GitError → warning appended."""
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return _GitError("log failed")
        if args[:2] == ["status", "--porcelain"]:
            return ""
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        return _GitError("error")

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    context = get_git_context()
    assert context is not None
    assert "_git_warnings" in context
    assert any("head info" in w for w in context["_git_warnings"])


def test_get_git_context_dirty_detect_returns_false(monkeypatch, tmp_path):
    """Cover branch [819,821]: _detect_dirty_status returns False directly."""
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)

    def fake_run(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(repo_root_path)
        if args[:2] == ["log", "-1"]:
            return "abc1234\n" + ("a" * 40) + "\nHEAD -> main"
        return _GitError("unexpected")

    def fake_rc(args, timeout=2.0, repo_root=None):
        _ = timeout
        _ = repo_root
        # diff-files clean, diff-index clean, ls-files no untracked (rc=1)
        if args[:1] == ["diff-files"]:
            return 0
        if args[:1] == ["diff-index"]:
            return 0
        if args[:1] == ["ls-files"]:
            return 1  # rc=1 means no untracked → False
        return 0

    repo_root_path = repo_root.resolve()
    monkeypatch.setattr(git_context_module, "_run_git_command", fake_run)
    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)

    context = get_git_context()
    assert context is not None
    assert context["dirty"] is False


def test_run_git_command_timeout(monkeypatch):
    """Cover L71-72: subprocess.TimeoutExpired → _GitError('timeout')."""

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = git_context_module._run_git_command(["status"])
    assert isinstance(result, _GitError)
    assert result.reason == "timeout"


def test_run_git_command_file_not_found(monkeypatch):
    """Cover L73-74: FileNotFoundError → _GitError('git not found')."""

    def _fake_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = git_context_module._run_git_command(["status"])
    assert isinstance(result, _GitError)
    assert result.reason == "git not found"


def test_run_git_command_os_error(monkeypatch):
    """Cover L75-76: OSError → _GitError('OSError: ...')."""

    def _fake_run(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = git_context_module._run_git_command(["status"])
    assert isinstance(result, _GitError)
    assert "OSError" in result.reason


def test_run_git_command_rc_timeout(monkeypatch):
    """Cover L98-99: _run_git_command_rc timeout."""

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = git_context_module._run_git_command_rc(["status"])
    assert isinstance(result, _GitError)
    assert result.reason == "timeout"


def test_run_git_command_rc_file_not_found(monkeypatch):
    """Cover L100-101: _run_git_command_rc git not found."""

    def _fake_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = git_context_module._run_git_command_rc(["status"])
    assert isinstance(result, _GitError)
    assert result.reason == "git not found"


def test_run_git_command_rc_os_error(monkeypatch):
    """Cover L102-103: _run_git_command_rc OSError."""

    def _fake_run(*args, **kwargs):
        raise OSError("broken pipe")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = git_context_module._run_git_command_rc(["status"])
    assert isinstance(result, _GitError)
    assert "OSError" in result.reason


def test_resolve_cwd_os_error(monkeypatch):
    """Cover L109-110: _resolve_cwd fails with OSError → None."""
    monkeypatch.setattr(
        Path, "cwd", classmethod(lambda cls: (_ for _ in ()).throw(OSError("no cwd")))
    )
    result = git_context_module._resolve_cwd()
    assert result is None


def test_resolve_git_dir_empty_git_file(tmp_path):
    """Cover L130-131: .git file is empty → None."""
    git_file = tmp_path / ".git"
    git_file.write_text("", encoding="utf-8")
    result = git_context_module._resolve_git_dir(str(tmp_path))
    assert result is None


def test_resolve_git_dir_no_gitdir_prefix(tmp_path):
    """Cover L133-134: .git file without 'gitdir:' prefix → None."""
    git_file = tmp_path / ".git"
    git_file.write_text("something else entirely", encoding="utf-8")
    result = git_context_module._resolve_git_dir(str(tmp_path))
    assert result is None


def test_resolve_git_dir_empty_target_path(tmp_path):
    """Cover L136-137: .git file with 'gitdir:' but empty path → None."""
    git_file = tmp_path / ".git"
    git_file.write_text("gitdir:   ", encoding="utf-8")
    result = git_context_module._resolve_git_dir(str(tmp_path))
    assert result is None


def test_resolve_git_dir_relative_gitdir(tmp_path):
    """Cover L140-141: .git file with relative gitdir path."""
    actual_git = tmp_path / "worktrees" / "main"
    actual_git.mkdir(parents=True)
    git_file = tmp_path / ".git"
    git_file.write_text("gitdir: worktrees/main", encoding="utf-8")
    result = git_context_module._resolve_git_dir(str(tmp_path))
    assert result == actual_git.resolve()


def test_extract_crash_frame_bad_index():
    """Cover L340-341, L345-347: crash_frame_index is invalid string."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": "not-a-number",
        "frames": [{"file_rel": "app.py", "line": 1}],
    }
    frame = git_context_module._extract_crash_frame(snapshot)
    # Falls back to index 0
    assert frame is not None
    assert frame["file_rel"] == "app.py"


def test_extract_crash_frame_out_of_range():
    """Cover L345-346: crash_frame_index out of range → fallback to [0]."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 99,
        "frames": [{"file_rel": "app.py", "line": 1}],
    }
    frame = git_context_module._extract_crash_frame(snapshot)
    assert frame is not None
    assert frame["file_rel"] == "app.py"


def test_extract_crash_line_string():
    """Cover L354-356: line is a string that can be parsed to int."""
    frame: dict[str, Any] = {"line": "42"}
    result = git_context_module._extract_crash_line(frame)
    assert result == 42


def test_extract_crash_line_invalid_string():
    """Cover L357-358: line is a non-numeric string → None."""
    frame: dict[str, Any] = {"line": "not-a-number"}
    result = git_context_module._extract_crash_line(frame)
    assert result is None


def test_extract_crash_line_none():
    """Cover L359-360: line is neither int nor str → None."""
    frame: dict[str, Any] = {"line": [1, 2]}
    result = git_context_module._extract_crash_line(frame)
    assert result is None


def test_resolve_repo_relative_file_dotdot():
    """Cover L381: file_rel starts with '..' → returns (rel_candidate, False)."""
    frame: dict[str, Any] = {"file_rel": "../outside.py"}
    rel, in_repo = git_context_module._resolve_repo_relative_file(frame, repo_root="/repo")
    assert rel == "../outside.py"
    assert in_repo is False


def test_resolve_repo_relative_file_abs_outside_repo():
    """Cover L389-390: abs file outside repo → (raw, False)."""
    frame: dict[str, Any] = {"file": "/other/place/app.py"}
    _rel, in_repo = git_context_module._resolve_repo_relative_file(frame, repo_root="/repo")
    assert in_repo is False


def test_resolve_repo_relative_file_no_file():
    """Cover L392: no file or file_rel → (None, False)."""
    frame: dict[str, Any] = {"line": 1}
    rel, in_repo = git_context_module._resolve_repo_relative_file(frame, repo_root="/repo")
    assert rel is None
    assert in_repo is False


def test_parse_blame_porcelain_empty():
    """Cover L396-397: empty blame text → None."""
    result = git_context_module._parse_blame_porcelain("", 10)
    assert result is None


def test_parse_blame_porcelain_no_commit():
    """Cover L428-429: blame output without commit line → None."""
    text = "author Alice\nsummary Fix bug\n"
    result = git_context_module._parse_blame_porcelain(text, 10)
    assert result is None


def test_parse_blame_porcelain_invalid_author_time():
    """Cover L420-421: non-integer author-time → stored as raw string."""
    text = (
        "def56789abcdef0123456789abcdef0123456789 10 10 1\n"
        "author Alice\n"
        "author-time not-a-timestamp\n"
        "summary Fix bug\n"
    )
    result = git_context_module._parse_blame_porcelain(text, 10)
    assert result is not None
    assert result["author_time"] == "not-a-timestamp"


def test_parse_shortstat_no_match():
    """Cover L444-445: shortstat line that doesn't match regex."""
    result = git_context_module._parse_shortstat("no match here")
    assert result == (0, 0, 0)


def test_parse_numstat_empty_lines():
    """Cover L508-509: numstat with empty lines and short parts."""
    text = "\n\nbinary\t\n5\t3\tfile.py\n"
    result = git_context_module._parse_numstat(text)
    assert result["files_changed"] >= 1


def test_detect_dirty_status_unstaged_error(monkeypatch):
    """Cover L197-198: diff-files returns _GitError → None."""
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command_rc",
        lambda args, repo_root=None, timeout=2.0: _GitError("fail"),
    )
    result = git_context_module._detect_dirty_status("/repo")
    assert result is None


def test_detect_dirty_status_unstaged_unexpected_rc(monkeypatch):
    """Cover L201-202: diff-files returns unexpected exit code (not 0 or 1)."""
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command_rc",
        lambda args, repo_root=None, timeout=2.0: 128,
    )
    result = git_context_module._detect_dirty_status("/repo")
    assert result is None


def test_detect_dirty_status_staged_error(monkeypatch):
    """Cover L208-209: diff-index returns _GitError → None."""
    call_num = {"n": 0}

    def fake_rc(args, repo_root=None, timeout=2.0):
        call_num["n"] += 1
        if call_num["n"] == 1:
            return 0  # diff-files ok
        return _GitError("fail")  # diff-index fails

    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)
    result = git_context_module._detect_dirty_status("/repo")
    assert result is None


def test_detect_dirty_status_staged_dirty(monkeypatch):
    """Cover L210-211: diff-index returns 1 → True (staged changes)."""
    call_num = {"n": 0}

    def fake_rc(args, repo_root=None, timeout=2.0):
        call_num["n"] += 1
        if call_num["n"] == 1:
            return 0  # diff-files clean
        return 1  # diff-index dirty

    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)
    result = git_context_module._detect_dirty_status("/repo")
    assert result is True


def test_detect_dirty_status_staged_unexpected_rc(monkeypatch):
    """Cover L214: diff-index returns unexpected exit code → None."""
    call_num = {"n": 0}

    def fake_rc(args, repo_root=None, timeout=2.0):
        call_num["n"] += 1
        if call_num["n"] == 1:
            return 0  # diff-files clean
        return 128  # diff-index unexpected

    monkeypatch.setattr(git_context_module, "_run_git_command_rc", fake_rc)
    result = git_context_module._detect_dirty_status("/repo")
    assert result is None


def test_detect_untracked_error(monkeypatch):
    """Cover L182-183: ls-files returns _GitError → None."""
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command_rc",
        lambda args, repo_root=None, timeout=2.0: _GitError("fail"),
    )
    result = git_context_module._detect_untracked_status("/repo")
    assert result is None


def test_detect_untracked_unexpected_rc(monkeypatch):
    """Cover L187-188: ls-files returns unexpected exit code → None."""
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command_rc",
        lambda args, repo_root=None, timeout=2.0: 128,
    )
    result = git_context_module._detect_untracked_status("/repo")
    assert result is None


def test_hydrate_context_with_dirty_fallback_porcelain(monkeypatch):
    """Cover L223-226: _detect_dirty_status None → fall back to porcelain."""
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command_rc",
        lambda args, repo_root=None, timeout=2.0: _GitError("fail"),
    )
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command",
        lambda args, repo_root=None, timeout=2.0: "M file.py",
    )
    result = git_context_module._hydrate_context_with_dirty({"commit": "abc"}, repo_root="/repo")
    assert result["dirty"] is True


def test_hydrate_context_both_fail(monkeypatch):
    """Cover L227-229: both _detect and porcelain fail → no dirty key."""
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command_rc",
        lambda args, repo_root=None, timeout=2.0: _GitError("fail"),
    )
    monkeypatch.setattr(
        git_context_module,
        "_run_git_command",
        lambda args, repo_root=None, timeout=2.0: _GitError("fail"),
    )
    result = git_context_module._hydrate_context_with_dirty({"commit": "abc"}, repo_root="/repo")
    assert "dirty" not in result


def test_load_cached_none_repo_different_cwd(monkeypatch, tmp_path):
    """Cover L241-243: cached with repo_root=None, different cwd → miss."""
    import time as _time

    entry = git_context_module._GitContextCacheEntry(
        expires_at=_time.monotonic() + 60,
        cwd="/some/other/cwd",
        repo_root=None,
        git_dir=None,
        head_mtime_ns=None,
        index_mtime_ns=None,
        dirty_cached=None,
        context_static=None,
    )
    git_context_module._git_context_cache = entry
    monkeypatch.chdir(tmp_path)
    hit, _ctx = git_context_module._load_cached_git_context(dirty_mode="always")
    assert hit is False


def test_load_cached_cwd_outside_repo(monkeypatch, tmp_path):
    """Cover L245-246: cwd outside cached repo_root → cache miss."""
    import time as _time

    entry = git_context_module._GitContextCacheEntry(
        expires_at=_time.monotonic() + 60,
        cwd=str(tmp_path),
        repo_root="/nonexistent/repo",
        git_dir=None,
        head_mtime_ns=None,
        index_mtime_ns=None,
        dirty_cached=None,
        context_static=None,
    )
    git_context_module._git_context_cache = entry
    monkeypatch.chdir(tmp_path)
    hit, _ctx = git_context_module._load_cached_git_context(dirty_mode="always")
    assert hit is False


def test_load_cached_context_static_none(monkeypatch, tmp_path):
    """Cover L256-257: context_static=None with valid repo → (True, None)."""
    import time as _time

    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"")
    monkeypatch.chdir(repo_root)

    resolved_git_dir = str(git_dir.resolve())
    head_mtime = git_dir.joinpath("HEAD").stat().st_mtime_ns
    index_mtime = git_dir.joinpath("index").stat().st_mtime_ns

    entry = git_context_module._GitContextCacheEntry(
        expires_at=_time.monotonic() + 60,
        cwd=str(repo_root.resolve()),
        repo_root=str(repo_root.resolve()),
        git_dir=resolved_git_dir,
        head_mtime_ns=head_mtime,
        index_mtime_ns=index_mtime,
        dirty_cached=None,
        context_static=None,
    )
    git_context_module._git_context_cache = entry
    hit, ctx = git_context_module._load_cached_git_context(dirty_mode="always")
    assert hit is True
    assert ctx is None


def test_discover_repo_root_all_fail(monkeypatch):
    """Cover L633-639: all candidates fail → warning with preview."""
    snapshot: dict[str, Any] = {
        "crash_frame_index": 0,
        "frames": [{"file_rel": "app.py", "line": 1}],
    }
    warnings: list[str] = []

    monkeypatch.setattr(
        git_context_module,
        "_run_git_command",
        lambda args, timeout=2.0, repo_root=None: _GitError("not a git repo"),
    )
    import time as _time

    result = git_context_module._discover_repo_root(
        snapshot,
        out_dir="/fake/dir",
        warnings=warnings,
        per_cmd_timeout_sec=1.0,
        deadline=_time.monotonic() + 5.0,
    )
    assert result is None
    assert any("not found from candidates" in w for w in warnings)


def test_run_git_budgeted_budget_exhausted(monkeypatch):
    """Cover L320-321: budget already exhausted → warning."""
    import time as _time

    warnings: list[str] = []
    result = git_context_module._run_git_budgeted(
        ["status"],
        repo_root=None,
        label="test",
        warnings=warnings,
        per_cmd_timeout_sec=1.0,
        deadline=_time.monotonic() - 1.0,  # already expired
    )
    assert result is None
    assert any("budget exhausted" in w for w in warnings)
