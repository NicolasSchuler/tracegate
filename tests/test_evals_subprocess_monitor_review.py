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


def test_orc03_drain_ends_while_a_grandchild_keeps_writing() -> None:
    """codex 1e#4: the drain bound is wall-clock, not "until the queue is empty"."""
    started = time.perf_counter()
    result = run_monitored_subprocess(
        cmd=[
            sys.executable,
            "-c",
            "import subprocess, sys\n"
            "subprocess.Popen(\n"
            "    [sys.executable, '-u', '-c',\n"
            "     \"import time\\nwhile True:\\n    print('grandchild chatter')\\n"
            '    time.sleep(0.005)\\n"]\n'
            ")\n"
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


def test_orc03_slow_progress_callbacks_do_not_extend_the_drain() -> None:
    """codex 1e#4: a slow progress callback must not push the drain past its bound."""
    calls: list[str] = []

    def _slow_callback(event: str, payload: dict[str, object] | None) -> None:
        del payload
        calls.append(event)
        time.sleep(0.3)

    started = time.perf_counter()
    result = run_monitored_subprocess(
        cmd=[
            sys.executable,
            "-c",
            "import subprocess, sys\n"
            "subprocess.Popen(\n"
            "    [sys.executable, '-u', '-c',\n"
            '     "import itertools, time\\nfor i in itertools.count():\\n'
            "    print('chatter', i)\\n    time.sleep(0.005)\\n\"]\n"
            ")\n"
            "print('parent done')\n",
        ],
        cwd=None,
        env=None,
        timeout_sec=30,
        progress_callback=_slow_callback,
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 0
    assert "parent done" in result.stdout
    assert elapsed < DRAIN_GRACE_SEC + 5
