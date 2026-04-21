"""Tests for the command-based patcher (argv and stdin modes)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from evals.patchers.command_patcher import CommandPatcher, patcher_from_env


def _python_cmd(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


def test_command_patcher_argv_mode_success(tmp_path: Path) -> None:
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    diff_text = "diff --git a/demo.py b/demo.py\\n"
    patcher = CommandPatcher(
        cmd=_python_cmd(f"import sys; sys.stdout.write({diff_text!r})"),
        shell=False,
        timeout_sec=5.0,
    )

    result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=tmp_path)

    assert result.diff == diff_text.strip()
    assert result.meta is not None
    assert result.meta.get("returncode") == 0
    assert result.meta.get("timed_out") is False
    assert float(result.meta.get("timeout_sec") or 0.0) == 5.0


def test_command_patcher_timeout_returns_structured_error(tmp_path: Path) -> None:
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(
        cmd=_python_cmd("import time; time.sleep(0.5)"),
        shell=False,
        timeout_sec=0.05,
    )

    result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta.get("error") == "timeout"
    assert result.meta.get("timed_out") is True
    assert float(result.meta.get("timeout_sec") or 0.0) == 0.05


def test_command_patcher_shell_mode_expands_environment(tmp_path: Path) -> None:
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(
        cmd="echo $LLMDEBUG_EVAL_CONTEXT_DIR",
        shell=True,
        timeout_sec=2.0,
    )

    result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=tmp_path)

    assert result.meta is not None
    assert result.meta.get("returncode") == 0
    assert result.meta.get("timed_out") is False
    assert result.diff == str(context_dir)


def test_command_patcher_writes_prompt_file_for_subprocess(tmp_path: Path) -> None:
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    prompt_text = "Prompt payload for subprocess"
    patcher = CommandPatcher(
        cmd=_python_cmd(
            "import os, pathlib, sys; "
            "path = pathlib.Path(os.environ['LLMDEBUG_EVAL_PROMPT_FILE']); "
            "sys.stdout.write(path.read_text(encoding='utf-8'))"
        ),
        shell=False,
        timeout_sec=2.0,
    )

    result = patcher.propose_patch(prompt=prompt_text, context_dir=context_dir, cwd=tmp_path)

    assert result.meta is not None
    assert result.meta.get("returncode") == 0
    assert result.diff == prompt_text
    assert (context_dir / "prompt.txt").read_text(encoding="utf-8") == prompt_text


def test_command_patcher_missing_binary_returns_structured_error(tmp_path: Path) -> None:
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(
        cmd="definitely-not-a-real-command --version",
        shell=False,
        timeout_sec=2.0,
    )

    result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta.get("error") == "spawn_failed"
    assert result.meta.get("timed_out") is False
    assert "definitely-not-a-real-command" in str(result.meta.get("stderr") or "")


def test_patcher_from_env_reads_shell_and_timeout(monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD", "echo hello")
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", "true")
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", "12.5")

    patcher = patcher_from_env()

    assert patcher is not None
    assert patcher.cmd == "echo hello"
    assert patcher.shell is True
    assert patcher.timeout_sec == 12.5


# ---------------------------------------------------------------------------
# Coverage gap: lines 91-93 — OSError writing prompt file
# ---------------------------------------------------------------------------
def test_prompt_file_write_oserror(tmp_path: Path, monkeypatch) -> None:
    """When the prompt file cannot be written, return a structured error."""
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(cmd="true", shell=False, timeout_sec=2.0)

    # Make write_text raise an OSError
    def _fail_write(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _fail_write)

    result = patcher.propose_patch(prompt="hi", context_dir=context_dir, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta.get("error") == "prompt_file_write_failed"
    assert "disk full" in str(result.meta.get("stderr", ""))
    assert result.meta.get("timed_out") is False


# ---------------------------------------------------------------------------
# Coverage gap: lines 113-114 — shlex.split raises ValueError (unclosed quote)
# ---------------------------------------------------------------------------
def test_invalid_shlex_cmd_returns_error(tmp_path: Path) -> None:
    """An unterminated quote in cmd causes an invalid_patch_cmd error."""
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(cmd="echo 'unterminated", shell=False, timeout_sec=2.0)

    result = patcher.propose_patch(prompt="hi", context_dir=context_dir, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta.get("error") == "invalid_patch_cmd"
    assert "ValueError" in str(result.meta.get("stderr", ""))


# ---------------------------------------------------------------------------
# Coverage gap: line 126 — shlex.split produces empty argv
# ---------------------------------------------------------------------------
def test_empty_cmd_after_shlex_split_returns_error(tmp_path: Path) -> None:
    """A cmd that resolves to no tokens returns an empty_patch_cmd error."""
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    # Empty string produces an empty argv list via shlex.split
    patcher = CommandPatcher(cmd="", shell=False, timeout_sec=2.0)

    result = patcher.propose_patch(prompt="hi", context_dir=context_dir, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta.get("error") == "empty_patch_cmd"


# ---------------------------------------------------------------------------
# Coverage gap: empty stdout from subprocess → diff is None
# ---------------------------------------------------------------------------
def test_empty_stdout_returns_none_diff(tmp_path: Path) -> None:
    """When the subprocess prints nothing, diff should be None."""
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(
        cmd=_python_cmd("pass"),  # prints nothing
        shell=False,
        timeout_sec=2.0,
    )

    result = patcher.propose_patch(prompt="hi", context_dir=context_dir, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta.get("returncode") == 0
    assert result.meta.get("timed_out") is False


# ---------------------------------------------------------------------------
# Coverage gap: subprocess returns non-zero exit code
# ---------------------------------------------------------------------------
def test_nonzero_exit_code_still_captures_output(tmp_path: Path) -> None:
    """A non-zero exit code is recorded in meta; diff may still be captured."""
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    patcher = CommandPatcher(
        cmd=_python_cmd("import sys; sys.stdout.write('partial diff'); sys.exit(1)"),
        shell=False,
        timeout_sec=2.0,
    )

    result = patcher.propose_patch(prompt="hi", context_dir=context_dir, cwd=tmp_path)

    assert result.meta is not None
    assert result.meta.get("returncode") == 1
    assert result.diff == "partial diff"


# ---------------------------------------------------------------------------
# Coverage gap: lines 198, 205, 208-209 — _env_bool / _env_float helpers
# ---------------------------------------------------------------------------
def test_patcher_from_env_returns_none_when_cmd_unset(monkeypatch) -> None:
    monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD", raising=False)
    assert patcher_from_env() is None


def test_patcher_from_env_returns_none_when_cmd_blank(monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD", "  ")
    assert patcher_from_env() is None


def test_env_float_invalid_falls_back_to_default(monkeypatch) -> None:
    """Non-numeric timeout string falls back to the default 30.0."""
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD", "echo ok")
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", "not-a-number")
    monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", raising=False)

    patcher = patcher_from_env()

    assert patcher is not None
    assert patcher.timeout_sec == 30.0


def test_env_bool_false_variants(monkeypatch) -> None:
    """_env_bool returns False for non-truthy strings."""
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD", "echo ok")
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", "no")
    monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", raising=False)

    patcher = patcher_from_env()

    assert patcher is not None
    assert patcher.shell is False


# ---------------------------------------------------------------------------
# Coverage gap: line 229 — patcher_from_env when LLMDEBUG_EVAL_PATCH_CMD is empty
# ---------------------------------------------------------------------------
def test_patcher_from_env_defaults(monkeypatch) -> None:
    """patcher_from_env uses defaults when optional env vars are absent."""
    monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD", "my-cmd")
    monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", raising=False)

    patcher = patcher_from_env()

    assert patcher is not None
    assert patcher.cmd == "my-cmd"
    assert patcher.shell is False
    assert patcher.timeout_sec == 30.0
