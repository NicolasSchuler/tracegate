"""Subprocess monitor regressions from the 2026-10-04 review (ORC-03, ORC-18, ORC-19)."""

from __future__ import annotations

import sys
import time

from baselines.subprocess_monitor import DRAIN_GRACE_SEC, run_monitored_subprocess

_SPAWN_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "time.sleep(60)\n"
)


def test_orc03_timeout_with_grandchild_returns_within_timeout_plus_grace() -> None:
    started = time.perf_counter()
    result = run_monitored_subprocess(
        cmd=[sys.executable, "-c", _SPAWN_GRANDCHILD], cwd=None, env=None, timeout_sec=2
    )
    elapsed = time.perf_counter() - started
    assert result.timed_out is True
    assert elapsed < 2 + DRAIN_GRACE_SEC + 3


def test_orc03_child_exit_with_lingering_grandchild_does_not_hang() -> None:
    started = time.perf_counter()
    result = run_monitored_subprocess(
        cmd=[
            sys.executable,
            "-c",
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print('parent done')\n",
        ],
        cwd=None,
        env=None,
        timeout_sec=30,
    )
    elapsed = time.perf_counter() - started
    assert result.timed_out is False
    assert result.returncode == 0
    assert "parent done" in result.stdout
    assert elapsed < DRAIN_GRACE_SEC + 5


def test_orc18_zero_timeout_means_no_deadline() -> None:
    result = run_monitored_subprocess(
        cmd=[sys.executable, "-c", "import time; time.sleep(0.5); print('ok')"],
        cwd=None,
        env=None,
        timeout_sec=0,
    )
    assert result.timed_out is False
    assert result.stdout.strip() == "ok"


def test_orc19_child_does_not_inherit_stdin() -> None:
    result = run_monitored_subprocess(
        cmd=[sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
        cwd=None,
        env=None,
        timeout_sec=10,
    )
    assert result.stdout.strip() == "''"
