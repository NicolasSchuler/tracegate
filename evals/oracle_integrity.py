"""Keep the pass/fail oracle out of the model's reach (review finding ATT-01, decision D10).

A model patch could edit ``test_buggy.py`` or add a ``conftest.py`` hook (for example one
that sets ``session.exitstatus = 0``) and turn a failing run into a "success". Before
every verification run the harness therefore restores the pristine test surface from
the case directory, and it decides success from pytest's JUnit counts instead of the
process exit code alone.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TEST_SURFACE_FILENAMES = frozenset(
    {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
)
"""Files that configure or hook pytest collection and reporting."""

JUNIT_REPORT_NAME = ".tracegate_pytest_verdict.xml"
"""Relative to the workdir, so it resolves the same way under every executor."""

_IGNORED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def is_test_surface_path(relative: Path) -> bool:
    """Return whether a workdir-relative path belongs to the test oracle."""
    name = relative.name
    if name in TEST_SURFACE_FILENAMES:
        return True
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _test_surface_files(root: Path) -> dict[Path, Path]:
    found: dict[Path, Path] = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIR_NAMES for part in relative.parts):
            continue
        if path.is_file() and is_test_surface_path(relative):
            found[relative] = path
    return found


def restore_test_surface(*, workdir: Path, pristine_dir: Path) -> tuple[str, ...]:
    """Make the workdir's test surface identical to the pristine case directory.

    Modified files are restored, files the patch added are deleted and deleted files are
    recreated. Returns the sorted workdir-relative paths that had been changed.
    """
    current = _test_surface_files(workdir)
    pristine = _test_surface_files(pristine_dir)
    changed: list[str] = []
    for relative, path in current.items():
        source = pristine.get(relative)
        if source is None:
            path.unlink()
            changed.append(relative.as_posix())
        elif path.read_bytes() != source.read_bytes():
            shutil.copyfile(source, path)
            changed.append(relative.as_posix())
    for relative, source in pristine.items():
        if relative not in current:
            target = workdir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            changed.append(relative.as_posix())
    return tuple(sorted(changed))


_JUNIT_FLAG = f"--junitxml={JUNIT_REPORT_NAME}"


def with_junit_report(python_args: Sequence[str]) -> list[str]:
    """Append the harness JUnit report flag to a ``python -m pytest`` argument list."""
    return [*python_args, _JUNIT_FLAG]


def wants_junit_report(python_args: Sequence[str]) -> bool:
    return _JUNIT_FLAG in python_args


@dataclass(frozen=True)
class PytestCounts:
    """Test counts from a JUnit report."""

    tests: int
    failures: int
    errors: int
    skipped: int

    @property
    def executed(self) -> int:
        return self.tests - self.skipped


def read_junit_counts(path: Path) -> PytestCounts | None:
    """Sum the counts over every ``<testsuite>`` in a JUnit report, or ``None``."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    # pytest never writes a DTD; refusing one rules out entity-expansion payloads.
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        return None
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return None
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    try:
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, "0"))
    except ValueError:
        return None
    return PytestCounts(**totals)


def consume_junit_counts(workdir: Path) -> PytestCounts | None:
    """Read and delete the workdir's JUnit report, so later prompts never see it."""
    path = workdir / JUNIT_REPORT_NAME
    counts = read_junit_counts(path)
    path.unlink(missing_ok=True)
    return counts


def run_succeeded(*, returncode: int, junit_requested: bool, counts: PytestCounts | None) -> bool:
    """Verdict for a verification run.

    Runs that requested the harness report need it (:func:`verified_success`). Results
    built without one (stubbed executors in tests) fall back to the exit code.
    """
    if junit_requested:
        return verified_success(returncode=returncode, counts=counts)
    return returncode == 0


def verified_success(*, returncode: int, counts: PytestCounts | None) -> bool:
    """Success needs exit code 0 and a report with at least one executed test, no failures
    and no errors. A missing report (the process died or never wrote one) is a failure."""
    if returncode != 0 or counts is None:
        return False
    return counts.executed >= 1 and counts.failures == 0 and counts.errors == 0
