"""Keep the pass/fail oracle out of the model's reach (review finding ATT-01, decision D10).

A model patch could edit ``test_buggy.py`` or add a ``conftest.py`` hook (for example one
that sets ``session.exitstatus = 0``) and turn a failing run into a "success". Before
every verification run the harness therefore restores the pristine test surface from
the case directory, and it decides success from pytest's JUnit counts instead of the
process exit code alone.

Threat model: this closes the realistic reward-hacking routes (edited tests, pytest
config and hook files, modules that shadow ``pytest``, symlinks, stale reports). The
patched code still runs inside the pytest process, so a deliberately adversarial patch
could forge a verdict; that would need a separate trusted reporting process.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TEST_SURFACE_FILENAMES = frozenset(
    {
        # pytest configuration and hooks
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        # modules that shadow pytest on ``python -m pytest`` or run at interpreter start
        "pytest.py",
        "py.py",
        "sitecustomize.py",
        "usercustomize.py",
    }
)
"""Files that configure, hook or replace pytest collection and reporting."""

_TEST_SURFACE_DIR_NAMES = frozenset({"pytest", "_pytest"})
"""Packages that would shadow pytest itself; everything below them is test surface."""

JUNIT_REPORT_PREFIX = ".tracegate_pytest_verdict"
"""Reports live in the workdir so the path resolves the same way under every executor."""

_IGNORED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})

ORACLE_POLICY_VERSION = 1
"""Behaviour version of the oracle policy below. Bump it when the policy changes."""


def oracle_policy_metadata() -> dict[str, object]:
    """The effective test-oracle policy, for ``patcher_config`` and the run fingerprint.

    The policy has no CLI knobs, so without this two runs whose success criteria differ
    (restored test surface, JUnit-counted verdict, surface rule set) would share a
    settings fingerprint and be indistinguishable in the records.
    """
    return {
        "version": int(ORACLE_POLICY_VERSION),
        "restore_test_surface_before_verification": True,
        "verdict_source": "junit_counts",
        "test_surface_filenames": sorted(TEST_SURFACE_FILENAMES),
        "test_surface_shadow_dirs": sorted(_TEST_SURFACE_DIR_NAMES),
    }


def is_test_surface_path(relative: Path) -> bool:
    """Return whether a workdir-relative path belongs to the test oracle."""
    if any(part in _TEST_SURFACE_DIR_NAMES for part in relative.parts[:-1]):
        return True
    name = relative.name
    if name in TEST_SURFACE_FILENAMES or name.endswith(".pth"):
        return True
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _test_surface_entries(root: Path) -> dict[Path, Path]:
    """Map relative test-surface paths to files or symlinks under *root*, never following links."""
    found: dict[Path, Path] = {}
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = [name for name in dirnames if name not in _IGNORED_DIR_NAMES]
        # Symlinked directories are entries too: they are removed, never traversed.
        for name in [*filenames, *(d for d in dirnames if (base / d).is_symlink())]:
            path = base / name
            relative = path.relative_to(root)
            if _is_hostile_entry(path) or (path.is_file() and is_test_surface_path(relative)):
                found[relative] = path
    return found


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _is_hostile_entry(path: Path) -> bool:
    """Symlinks and special files (FIFOs, devices, sockets) never belong in a workdir."""
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))


def _is_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _remove_entry(path: Path) -> None:
    """Clear *path* whatever it is: a file, a symlink, a special file or a directory.

    A patch can delete a pristine test file and create a directory in its place
    (``test_buggy.py/__init__.py``), so restoring the oracle has to be able to remove a
    directory too. Symlinks count as hostile entries and are unlinked, never followed, so
    nothing outside the workdir is reachable from here.
    """
    if _is_hostile_entry(path) or _is_regular_file(path):
        path.unlink()
    elif _is_directory(path):
        shutil.rmtree(path, ignore_errors=True)


def _clear_parent_obstructions(workdir: Path, relative: Path) -> None:
    """Remove the workdir parents of *relative* that are not plain directories."""
    for parent in reversed(relative.parents):
        candidate = workdir / parent
        if candidate == workdir or _is_directory(candidate):
            continue
        if candidate.is_symlink() or candidate.exists():
            candidate.unlink()


def _write_inside(workdir: Path, relative: Path, source: Path) -> None:
    target = workdir / relative
    _clear_parent_obstructions(workdir, target.relative_to(workdir))
    _remove_entry(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve().is_relative_to(workdir.resolve()):
        raise ValueError(f"refusing to restore outside the workdir: {relative}")
    shutil.copyfile(source, target)


def restore_test_surface(*, workdir: Path, pristine_dir: Path) -> tuple[str, ...]:
    """Make the workdir's test surface identical to the pristine case directory.

    Modified files are restored, files the patch added are deleted, deleted files are
    recreated and symlinks are removed rather than followed. Returns the sorted
    workdir-relative paths that had been changed.
    """
    current = _test_surface_entries(workdir)
    pristine = {
        relative: path
        for relative, path in _test_surface_entries(pristine_dir).items()
        if _is_regular_file(path)
    }
    changed: list[str] = []
    for relative, path in current.items():
        source = pristine.get(relative)
        if source is None:
            _remove_entry(path)
            changed.append(relative.as_posix())
        elif not _is_regular_file(path) or path.read_bytes() != source.read_bytes():
            _write_inside(workdir, relative, source)
            changed.append(relative.as_posix())
    for relative, source in pristine.items():
        if relative not in current:
            _write_inside(workdir, relative, source)
            changed.append(relative.as_posix())
    return tuple(sorted(changed))


_JUNIT_FLAG_PREFIX = f"--junitxml={JUNIT_REPORT_PREFIX}"


def with_junit_report(python_args: Sequence[str]) -> list[str]:
    """Append a JUnit report flag with a fresh per-run report name."""
    return [*python_args, f"{_JUNIT_FLAG_PREFIX}_{uuid.uuid4().hex}.xml"]


def junit_report_name(python_args: Sequence[str]) -> str | None:
    """Return the harness report name requested in *python_args*, if any."""
    for arg in python_args:
        if arg.startswith(_JUNIT_FLAG_PREFIX):
            name = arg.split("=", 1)[1]
            return name if "/" not in name and name.endswith(".xml") else None
    return None


def clear_junit_reports(workdir: Path) -> None:
    """Remove harness reports (and anything planted under their names) before a run."""
    if not workdir.is_dir():
        return
    for path in workdir.glob(f"{JUNIT_REPORT_PREFIX}*"):
        if stat.S_ISDIR(os.lstat(path).st_mode):
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()


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


def consume_junit_counts(workdir: Path, report_name: str) -> PytestCounts | None:
    """Read and delete a run's JUnit report, so later prompts never see it."""
    path = workdir / report_name
    counts = read_junit_counts(path) if _is_regular_file(path) else None
    clear_junit_reports(workdir)
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
