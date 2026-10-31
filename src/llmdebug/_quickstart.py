"""Offline quickstart snapshot generation.

This module deliberately has no CLI dependencies.  It creates the same
snapshot artifact as the public capture entry points while keeping the example
small, deterministic, and independent of repository or environment metadata.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from .capture import capture_exception
from .config import SnapshotConfig


@dataclass(frozen=True)
class QuickstartResult:
    """Paths and follow-up commands produced by the quickstart example.

    Attributes:
        snapshot_path: Path of the snapshot file written by the example.
        show_command: Shell-quoted ``llmdebug show`` command for the snapshot directory.
        list_command: Shell-quoted ``llmdebug list`` command for the snapshot directory.
    """

    snapshot_path: Path
    show_command: str
    list_command: str


def _calculate_average_latency() -> float:
    """Raise the deterministic failure used by the quickstart snapshot."""
    total_latency_ms = 120
    completed_requests = 0
    return total_latency_ms / completed_requests


def _next_command(command: str, out_dir: Path) -> str:
    """Return a copy-paste-safe command for the selected snapshot directory."""
    return shlex.join(("llmdebug", command, "--dir", str(out_dir)))


def create_quickstart_snapshot(out_dir: str | Path = ".llmdebug") -> QuickstartResult:
    """Create one real synthetic failure snapshot without external services.

    The snapshot captures a deterministic ``ZeroDivisionError`` with environment,
    git, coverage, async-context, and argument capture disabled, and without pruning
    older snapshots in the directory.

    Args:
        out_dir: Directory that receives the snapshot file.

    Returns:
        The written snapshot path and the follow-up ``show``/``list`` commands.

    Raises:
        RuntimeError: If the synthetic failure unexpectedly does not raise.
    """
    output_dir = Path(out_dir)
    config = SnapshotConfig(
        out_dir=str(output_dir),
        source_mode="crash_only",
        include_env=False,
        include_git=False,
        include_async_context=False,
        include_coverage=False,
        include_args=False,
        output_format="json",
        max_snapshots=0,
    )

    try:
        _calculate_average_latency()
    except ZeroDivisionError as exc:
        snapshot_path = capture_exception(
            "quickstart_example",
            exc,
            exc.__traceback__,
            config,
        )
        return QuickstartResult(
            snapshot_path=snapshot_path,
            show_command=_next_command("show", output_dir),
            list_command=_next_command("list", output_dir),
        )

    raise RuntimeError("quickstart example did not raise its expected synthetic failure")
