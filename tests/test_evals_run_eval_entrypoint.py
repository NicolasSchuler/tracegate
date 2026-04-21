"""Entrypoint smoke tests for evals.run_eval bootstrap behavior."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_module_entrypoint_help_succeeds() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "evals.run_eval", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Run the llmdebug multi-condition repair benchmark harness." in completed.stdout


def test_direct_script_entrypoint_help_succeeds() -> None:
    completed = subprocess.run(
        [sys.executable, "evals/run_eval.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Run the llmdebug multi-condition repair benchmark harness." in completed.stdout
