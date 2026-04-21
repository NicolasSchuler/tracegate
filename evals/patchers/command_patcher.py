"""Patcher that delegates patch generation to an external shell command.

``CommandPatcher`` spawns a subprocess (configured via :attr:`CommandPatcher.cmd`)
that is expected to print a unified diff to *stdout*.  The prompt text is passed
both on *stdin* and via the ``LLMDEBUG_EVAL_PROMPT_FILE`` environment variable;
the context directory is exposed as ``LLMDEBUG_EVAL_CONTEXT_DIR``.

This allows arbitrary external tools—scripts, other language models, or
purpose-built patch generators—to plug into the eval harness without requiring a
Python integration layer.

The convenience factory :func:`patcher_from_env` constructs a ``CommandPatcher``
from the ``LLMDEBUG_EVAL_PATCH_CMD`` family of environment variables.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .base import Patcher, PatchResult
from .payload_types import PatcherMeta


@dataclass(frozen=True)
class CommandPatcher(Patcher):
    """Run a local command that prints a unified diff to stdout.

    The command is invoked as a subprocess with the case's working directory set
    to *cwd*.  The debugging prompt is written to ``<context_dir>/prompt.txt``
    and passed on *stdin*; two environment variables are injected for the child
    process:

    * ``LLMDEBUG_EVAL_CONTEXT_DIR`` — absolute path to *context_dir*.
    * ``LLMDEBUG_EVAL_PROMPT_FILE`` — absolute path to the written prompt file.

    The child's *stdout* is captured and treated as the unified diff.  An empty
    or whitespace-only *stdout* is interpreted as "no patch produced".

    Attributes:
        cmd: Shell command string to execute.  When :attr:`shell` is ``False``
            (the default) the string is tokenised with :func:`shlex.split`;
            when ``True`` it is passed verbatim to the system shell.
        shell: Whether to run *cmd* through the system shell.  Defaults to
            ``False``.
        timeout_sec: Maximum wall-clock seconds to wait for the subprocess.
            Defaults to ``30.0``.
        name: Identifier string used in evaluation result metadata.
    """

    cmd: str
    shell: bool = False
    timeout_sec: float = 30.0
    name: str = "command"

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Invoke the external command and capture its stdout as a unified diff.

        Writes *prompt* to ``<context_dir>/prompt.txt``, sets the
        ``LLMDEBUG_EVAL_CONTEXT_DIR`` and ``LLMDEBUG_EVAL_PROMPT_FILE``
        environment variables, then runs :attr:`cmd` as a subprocess with
        *cwd* as the working directory.

        Args:
            prompt: The natural-language debugging prompt.  Passed both on
                *stdin* and via ``LLMDEBUG_EVAL_PROMPT_FILE``.
            context_dir: Directory used for per-case context artefacts.
                ``LLMDEBUG_EVAL_CONTEXT_DIR`` is set to its absolute path.
            cwd: Working directory for the subprocess; typically the root of
                the buggy case being evaluated.

        Returns:
            A :class:`~evals.patchers.base.PatchResult` whose ``diff`` field
            contains the stripped *stdout* of the command (or ``None`` if the
            output was empty).  On error the ``meta`` dict contains an
            ``"error"`` key describing the failure (e.g. ``"timeout"``,
            ``"spawn_failed"``, ``"invalid_patch_cmd"``).
        """
        timeout = max(0.01, float(self.timeout_sec))
        env = os.environ.copy()
        env["LLMDEBUG_EVAL_CONTEXT_DIR"] = str(context_dir)
        prompt_path = context_dir / "prompt.txt"
        env["LLMDEBUG_EVAL_PROMPT_FILE"] = str(prompt_path)
        try:
            context_dir.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
        except OSError as exc:
            message = f"{type(exc).__name__}: {exc}"
            return PatchResult(
                diff=None,
                meta=cast(
                    PatcherMeta,
                    {
                        "error": "prompt_file_write_failed",
                        "timed_out": False,
                        "timeout_sec": timeout,
                        "stderr": message,
                        "exc": message,
                        "prompt_file": str(prompt_path),
                    },
                ),
            )
        cmd: str | list[str]
        if self.shell:
            cmd = self.cmd
        else:
            try:
                cmd = shlex.split(self.cmd)
            except ValueError as exc:
                return PatchResult(
                    diff=None,
                    meta=cast(
                        PatcherMeta,
                        {
                            "error": "invalid_patch_cmd",
                            "stderr": f"{type(exc).__name__}: {exc}",
                            "timeout_sec": timeout,
                        },
                    ),
                )
            if not cmd:
                return PatchResult(
                    diff=None,
                    meta=cast(
                        PatcherMeta,
                        {
                            "error": "empty_patch_cmd",
                            "stderr": "Command resolved to no argv tokens.",
                            "timeout_sec": timeout,
                        },
                    ),
                )

        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                shell=self.shell,
                cwd=cwd,
                env=env,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
            return PatchResult(
                diff=None,
                meta=cast(
                    PatcherMeta,
                    {
                        "error": "timeout",
                        "timed_out": True,
                        "timeout_sec": timeout,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                ),
            )
        except OSError as exc:
            message = f"{type(exc).__name__}: {exc}"
            return PatchResult(
                diff=None,
                meta=cast(
                    PatcherMeta,
                    {
                        "error": "spawn_failed",
                        "timed_out": False,
                        "timeout_sec": timeout,
                        "stderr": message,
                        "exc": message,
                    },
                ),
            )

        diff = proc.stdout.strip() or None
        meta = cast(
            PatcherMeta,
            {
                "returncode": int(proc.returncode),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "timed_out": False,
                "timeout_sec": timeout,
            },
        )
        return PatchResult(diff=diff, meta=meta)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def patcher_from_env() -> CommandPatcher | None:
    """Build a :class:`CommandPatcher` from environment variables.

    Reads the following variables:

    * ``LLMDEBUG_EVAL_PATCH_CMD`` *(required)* — the command string to run.
    * ``LLMDEBUG_EVAL_PATCH_CMD_SHELL`` — ``"1"``/``"true"``/``"yes"``/``"on"``
      to enable shell mode; defaults to ``False``.
    * ``LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC`` — float timeout in seconds;
      defaults to ``30.0``.

    Returns:
        A configured :class:`CommandPatcher` instance, or ``None`` if
        ``LLMDEBUG_EVAL_PATCH_CMD`` is unset or empty.
    """
    cmd = os.getenv("LLMDEBUG_EVAL_PATCH_CMD", "").strip()
    if not cmd:
        return None
    shell = _env_bool("LLMDEBUG_EVAL_PATCH_CMD_SHELL", default=False)
    timeout_sec = _env_float("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", default=30.0)
    return CommandPatcher(cmd=cmd, shell=shell, timeout_sec=timeout_sec)
