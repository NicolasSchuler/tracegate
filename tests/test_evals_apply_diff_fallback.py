"""Tests for diff-application fallback when the primary patch tool fails."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from evals import run_eval


def _force_patch_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.shutil,
        "which",
        lambda name: "/usr/bin/patch" if name == "patch" else None,
    )

    def _failed_patch(*args, **kwargs):
        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "simulated hunk failure"

        return _Proc()

    monkeypatch.setattr(run_eval.subprocess, "run", _failed_patch)


def test_apply_diff_falls_back_to_relaxed_replacement(tmp_path, monkeypatch) -> None:
    _force_patch_failure(monkeypatch)

    target = tmp_path / "buggy.py"
    target.write_text("value = 1\n", encoding="utf-8")

    diff = """--- a/buggy.py
+++ b/buggy.py
@@ -99,1 +99,1 @@
-value = 1
+value = 2
"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert applied
    assert "relaxed replacement fallback" in message
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_apply_diff_falls_back_to_relaxed_multiline_replacement(tmp_path, monkeypatch) -> None:
    _force_patch_failure(monkeypatch)

    target = tmp_path / "buggy.py"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    diff = """--- a/buggy.py
+++ b/buggy.py
@@ -99,3 +99,4 @@
-alpha
-beta
+delta
+epsilon
+zeta
 gamma
"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert applied
    assert "relaxed replacement fallback" in message
    assert target.read_text(encoding="utf-8") == "delta\nepsilon\nzeta\ngamma\n"


def test_apply_diff_fallback_fails_when_target_missing(tmp_path, monkeypatch) -> None:
    _force_patch_failure(monkeypatch)

    target = tmp_path / "buggy.py"
    target.write_text("value = 1\n", encoding="utf-8")

    diff = """--- a/buggy.py
+++ b/buggy.py
@@ -99,1 +99,1 @@
-missing = 1
+value = 2
"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert not applied
    assert "relaxed fallback failed" in message
    assert "could not find context-anchored target" in message
    assert target.read_text(encoding="utf-8") == "value = 1\n"


def test_apply_diff_fallback_restores_original_after_partial_patch_write(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        run_eval.shutil,
        "which",
        lambda name: "/usr/bin/patch" if name == "patch" else None,
    )

    def _partially_failed_patch(*args, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "buggy.py").write_text("corrupted = 99\n", encoding="utf-8")

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "simulated partial write before failure"

        return _Proc()

    monkeypatch.setattr(run_eval.subprocess, "run", _partially_failed_patch)

    target = tmp_path / "buggy.py"
    target.write_text("value = 1\n", encoding="utf-8")

    diff = """--- a/buggy.py
+++ b/buggy.py
@@ -99,1 +99,1 @@
-value = 1
+value = 2
"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert applied
    assert "relaxed replacement fallback" in message
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_apply_diff_returns_timeout_when_patch_hangs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.shutil,
        "which",
        lambda name: "/usr/bin/patch" if name == "patch" else None,
    )

    def _timeout_patch(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=run_eval.PATCH_APPLY_TIMEOUT_SEC)

    monkeypatch.setattr(run_eval.subprocess, "run", _timeout_patch)

    target = tmp_path / "buggy.py"
    target.write_text("value = 1\n", encoding="utf-8")
    diff = """--- a/buggy.py
+++ b/buggy.py
@@ -1 +1 @@
-value = 1
+value = 2
"""

    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert not applied
    assert "patch timed out" in message


def test_apply_diff_rejects_path_traversal_before_patch_invocation(tmp_path, monkeypatch) -> None:
    def _unexpected_subprocess_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for unsafe diff paths")

    monkeypatch.setattr(run_eval.subprocess, "run", _unexpected_subprocess_run)

    diff = """diff --git a/../outside.txt b/../outside.txt
--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-outside
+pwned
"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert not applied
    assert (
        "patch rejected: unsafe path" in message
        or "patch rejected: path escapes workdir" in message
    )


def test_apply_diff_relaxed_replacement_is_transactional(tmp_path, monkeypatch) -> None:
    _force_patch_failure(monkeypatch)

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 1\n", encoding="utf-8")

    diff = """--- a/first.py
+++ b/first.py
@@ -99,1 +99,1 @@
-value = 1
+value = 2
--- a/second.py
+++ b/second.py
@@ -99,1 +99,1 @@
-missing = 1
+value = 2
"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert not applied
    assert "relaxed fallback failed" in message
    assert first.read_text(encoding="utf-8") == "value = 1\n"
    assert second.read_text(encoding="utf-8") == "value = 1\n"


def test_apply_diff_fallback_rejects_lossy_malformed_multiline_hunk(tmp_path, monkeypatch) -> None:
    _force_patch_failure(monkeypatch)

    target = tmp_path / "buggy.py"
    target.write_text(
        "fH) ZM9TB(rKr\nG+s\nmjQ8mT3OA\n94:HhblZa_QFPiyCE\n-5lk\n) nMLpQ-AK\nlXOELxyxeH8ilq",
        encoding="utf-8",
    )

    diff = """diff --git a/buggy.py b/buggy.py
--- a/buggy.py
+++ b/buggy.py
@@ -1,7 +1 @@
-fH) ZM9TB(rKr
-G+s
-mjQ8mT3OA
-94:HhblZa_QFPiyCE
--5lk
-) nMLpQ-AK
-lXOELxyxeH8ilq+-Y"""
    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert not applied
    assert message == "patch rejected: malformed unified diff hunk counts"
    assert target.read_text(encoding="utf-8") == (
        "fH) ZM9TB(rKr\nG+s\nmjQ8mT3OA\n94:HhblZa_QFPiyCE\n-5lk\n) nMLpQ-AK\nlXOELxyxeH8ilq"
    )


def test_apply_diff_patches_a_crlf_target_and_keeps_its_endings(tmp_path) -> None:
    """LLM-09: an LF model diff applies to a CRLF file, which keeps CRLF endings."""
    target = tmp_path / "buggy.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\n")
    diff = "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 3\n"

    ok, msg = run_eval.apply_diff(tmp_path, diff)

    assert ok, msg
    assert target.read_bytes() == b"a = 1\r\nb = 3\r\n"


def test_apply_diff_rolls_back_crlf_targets_when_one_cannot_be_prepared(tmp_path) -> None:
    """LLM-09 (codex 1g#5): a read-only second CRLF target leaves the first unchanged."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores the read-only bit")
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_bytes(b"x = 1\r\n")
    second.write_bytes(b"y = 1\r\n")
    second.chmod(stat.S_IRUSR)
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1,1 +1,1 @@\n-y = 1\n+y = 2\n"
    )
    try:
        ok, msg = run_eval.apply_diff(tmp_path, diff)
    finally:
        second.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert not ok
    assert msg.startswith("could not prepare CRLF diff targets")
    assert first.read_bytes() == b"x = 1\r\n"
    assert second.read_bytes() == b"y = 1\r\n"


def test_apply_diff_cleanup_keeps_named_orig_targets_and_drops_tool_leftovers(
    tmp_path, monkeypatch
) -> None:
    """ATT-16 (codex 1b#8): only ``.orig``/``.rej`` files the diff doesn't name are removed."""
    monkeypatch.setattr(
        run_eval.shutil, "which", lambda name: "/usr/bin/patch" if name == "patch" else None
    )

    def _patch_leaving_a_backup(*args, **kwargs):
        (tmp_path / "fixture.orig").write_text("expected = 1\n", encoding="utf-8")
        (tmp_path / "buggy.py.orig").write_text("backup\n", encoding="utf-8")

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Proc()

    monkeypatch.setattr(run_eval.subprocess, "run", _patch_leaving_a_backup)
    (tmp_path / "buggy.py").write_text("value = 1\n", encoding="utf-8")
    diff = "--- /dev/null\n+++ b/fixture.orig\n@@ -0,0 +1,1 @@\n+expected = 1\n"

    ok, msg = run_eval.apply_diff(tmp_path, diff)

    assert ok, msg
    assert (tmp_path / "fixture.orig").read_text(encoding="utf-8") == "expected = 1\n"
    assert not (tmp_path / "buggy.py.orig").exists()
