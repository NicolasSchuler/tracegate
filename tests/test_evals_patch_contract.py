"""Direct contract tests for eval patch parsing and verification helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evals import eval_patches as patches


def test_apply_diff_accepts_dev_null_new_and_deleted_file_headers(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "obsolete.py").write_text("OBSOLETE = True\n", encoding="utf-8")

    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
        "diff --git a/created.py b/created.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/created.py\n"
        "@@ -0,0 +1 @@\n"
        "+CREATED = True\n"
        "diff --git a/obsolete.py b/obsolete.py\n"
        "deleted file mode 100644\n"
        "--- a/obsolete.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-OBSOLETE = True\n"
    )

    applied, message = patches.apply_diff(tmp_path, diff)

    assert applied, message
    assert (tmp_path / "buggy.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (tmp_path / "created.py").read_text(encoding="utf-8") == "CREATED = True\n"
    assert not (tmp_path / "obsolete.py").exists()


def test_apply_diff_git_fallback_succeeds_when_patch_missing(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        patches.shutil,
        "which",
        lambda name: None if name == "patch" else ("/usr/bin/git" if name == "git" else None),
    )

    def _successful_git_apply(*args, **kwargs):
        calls.append(list(args[0]))

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Proc()

    monkeypatch.setattr(patches.subprocess, "run", _successful_git_apply)

    applied, message = patches.apply_diff(
        tmp_path,
        "diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert applied is True
    assert message == ""
    assert calls == [["/usr/bin/git", "apply", "--whitespace=nowarn", "-"]]


def test_apply_diff_git_fallback_surfaces_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        patches.shutil,
        "which",
        lambda name: None if name == "patch" else ("/usr/bin/git" if name == "git" else None),
    )

    def _failed_git_apply(*args, **kwargs):
        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "git rejected patch"

        return _Proc()

    monkeypatch.setattr(patches.subprocess, "run", _failed_git_apply)

    applied, message = patches.apply_diff(
        tmp_path,
        "diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert applied is False
    assert message == "git apply failed: git rejected patch"


def test_apply_diff_git_fallback_surfaces_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        patches.shutil,
        "which",
        lambda name: None if name == "patch" else ("/usr/bin/git" if name == "git" else None),
    )

    def _timeout_git_apply(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=patches.PATCH_APPLY_TIMEOUT_SEC)

    monkeypatch.setattr(patches.subprocess, "run", _timeout_git_apply)

    applied, message = patches.apply_diff(
        tmp_path,
        "diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert applied is False
    assert "git apply timed out" in message


def test_apply_diff_returns_error_when_no_patch_tool_is_available(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(patches.shutil, "which", lambda name: None)

    applied, message = patches.apply_diff(
        tmp_path,
        "diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert applied is False
    assert message == "No patch tool found (git or patch required)."


def test_extract_diff_paths_deduplicates_real_paths_and_ignores_dev_null() -> None:
    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
        "diff --git a/new_file.py b/new_file.py\n"
        "--- /dev/null\n"
        "+++ b/new_file.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )

    assert patches._extract_diff_paths(diff) == ["buggy.py", "new_file.py"]


def test_diff_target_helpers_deduplicate_repeated_file_entries(tmp_path: Path) -> None:
    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-b\n"
        "+c\n"
    )
    (tmp_path / "buggy.py").write_text("a\n", encoding="utf-8")

    assert patches._extract_diff_paths(diff) == ["buggy.py"]
    assert patches._backup_diff_target_files(cwd=tmp_path, diff=diff) == {"buggy.py": b"a\n"}
    assert patches._python_files_touched_by_diff(cwd=tmp_path, diff=diff) == ["buggy.py"]

    delete_missing = (
        "diff --git a/missing.py b/missing.py\n"
        "--- a/missing.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-gone\n"
    )
    patches._remove_deleted_diff_targets(cwd=tmp_path, diff=delete_missing)
    assert not (tmp_path / "missing.py").exists()


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("", None),
        ("a/", None),
        ("b/", None),
        ("a/buggy.py", "buggy.py"),
        ("/dev/null", "/dev/null"),
    ],
)
def test_normalize_diff_path_token(token: str, expected: str | None) -> None:
    assert patches._normalize_diff_path_token(token) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_buggy.py", True),
        ("pkg/conftest.py", True),
        ("pkg/test_buggy.py", True),
        ("pkg/buggy_test.py", True),
        ("pkg/buggy.py", False),
        ("", False),
    ],
)
def test_is_test_file_path(path: str, expected: bool) -> None:
    assert patches._is_test_file_path(path) is expected


def test_validate_diff_paths_allows_dev_null_only_in_file_headers(tmp_path: Path) -> None:
    ok, message = patches._validate_diff_paths_within_cwd(
        tmp_path,
        "diff --git a/buggy.py b/buggy.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/buggy.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n",
    )
    assert ok is True
    assert message == ""

    ok, message = patches._validate_diff_paths_within_cwd(
        tmp_path,
        "diff --git /dev/null b/buggy.py\n--- /dev/null\n+++ b/buggy.py\n@@ -0,0 +1 @@\n+x = 1\n",
    )

    assert ok is False
    assert message == "patch rejected: /dev/null is only supported in ---/+++ headers"


def test_validate_diff_paths_rejects_unsafe_escape(tmp_path: Path) -> None:
    ok, message = patches._validate_diff_paths_within_cwd(
        tmp_path,
        "diff --git a/../outside.py b/../outside.py\n"
        "--- a/../outside.py\n"
        "+++ b/../outside.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n",
    )

    assert ok is False
    assert "unsafe path" in message


def test_validate_diff_paths_rejects_missing_entries(tmp_path: Path) -> None:
    ok, message = patches._validate_diff_paths_within_cwd(tmp_path, "")

    assert ok is False
    assert message == "patch rejected: no file paths found in diff"


def test_validate_diff_paths_rejects_double_dev_null_headers(tmp_path: Path) -> None:
    ok, message = patches._validate_diff_paths_within_cwd(
        tmp_path,
        "diff --git a/buggy.py b/buggy.py\n--- /dev/null\n+++ /dev/null\n",
    )

    assert ok is False
    assert message == "patch rejected: invalid file header uses /dev/null on both sides"


def test_validate_diff_paths_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)

    ok, message = patches._validate_diff_paths_within_cwd(
        tmp_path,
        "diff --git a/linked/buggy.py b/linked/buggy.py\n"
        "--- a/linked/buggy.py\n"
        "+++ b/linked/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n",
    )

    assert ok is False
    assert "path escapes workdir" in message


def test_validate_unified_diff_structure_rejects_malformed_hunk_counts() -> None:
    ok, message = patches._validate_unified_diff_structure(
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1 @@\n"
        "-def compute() -> int:\n"
        "-    return 1+    return 2"
    )

    assert ok is False
    assert message == "patch rejected: malformed unified diff hunk counts"


def test_validate_unified_diff_structure_accepts_no_newline_markers() -> None:
    ok, message = patches._validate_unified_diff_structure(
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1 @@\n"
        "-before\n"
        "-after\n"
        "\\ No newline at end of file\n"
        "+updated\n"
        "\\ No newline at end of file\n"
    )

    assert ok is True
    assert message == ""


def test_run_syntax_preflight_passes_for_valid_python(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")

    passed, meta = patches._run_syntax_preflight(
        cwd=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert passed is True
    assert meta["syntax_preflight_checked_files"] == ["buggy.py"]
    assert meta["syntax_preflight_error_type"] == ""


def test_run_syntax_preflight_reports_pycompile_error(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    passed, meta = patches._run_syntax_preflight(
        cwd=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert passed is False
    assert meta["syntax_preflight_error_file"] == "buggy.py"
    assert meta["syntax_preflight_error_type"] == "PyCompileError"


def test_run_syntax_preflight_reports_generic_exception(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")

    def _raise_compile(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(patches.py_compile, "compile", _raise_compile)

    passed, meta = patches._run_syntax_preflight(
        cwd=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert passed is False
    assert meta["syntax_preflight_error_type"] == "RuntimeError"
    assert meta["syntax_preflight_error_msg"] == "RuntimeError: compiler unavailable"


def test_run_syntax_preflight_skips_when_no_python_files_are_touched(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")

    passed, meta = patches._run_syntax_preflight(
        cwd=tmp_path,
        diff="diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-hi\n+bye\n",
    )

    assert passed is True
    assert meta["syntax_preflight_checked_count"] == 0


def test_run_patch_sanitization_accepts_clean_patch() -> None:
    passed, meta = patches._run_patch_sanitization(
        diff=(
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1 +1 @@\n"
            "-return 1\n"
            "+return 2\n"
        )
    )

    assert passed is True
    assert meta["patch_sanitize_reasons"] == []


def test_run_patch_sanitization_rejects_malformed_patch_text() -> None:
    passed, meta = patches._run_patch_sanitization(
        diff=(
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1 +1 @@\n"
            "-raise Exception('boom')\n"
            "+raise\\Exception('boom')\n"
        )
    )

    assert passed is False
    assert meta["patch_sanitize_checked_files"] == ["buggy.py"]
    assert "Patch sanitation rejected malformed patch text" in meta["patch_sanitize_error_msg"]


def test_patch_sanitize_line_reasons_cover_escape_and_await_cases() -> None:
    assert "literal_escaped_newline_or_tab_in_code" in patches._patch_sanitize_line_reasons(
        "value = foo\\nbar"
    )
    assert patches._patch_sanitize_line_reasons('value = "foo\\nbar"') == []
    assert "suspicious_trailing_quote_comma_after_await" in patches._patch_sanitize_line_reasons(
        'await client.fetch("oops",'
    )


def test_backup_restore_and_python_file_discovery_cover_edge_cases(
    tmp_path: Path, monkeypatch
) -> None:
    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
        "diff --git a/new_file.py b/new_file.py\n"
        "--- /dev/null\n"
        "+++ b/new_file.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )
    (tmp_path / "buggy.py").write_text("a\n", encoding="utf-8")
    (tmp_path / "new_file.py").write_text("x = 1\n", encoding="utf-8")

    backups = patches._backup_diff_target_files(cwd=tmp_path, diff=diff)
    assert backups == {"buggy.py": b"a\n", "new_file.py": b"x = 1\n"}

    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError("boom")))
    assert patches._backup_diff_target_files(cwd=tmp_path, diff=diff) == {}

    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    monkeypatch.undo()
    patches._restore_diff_target_files(
        cwd=restore_dir,
        backups={"pkg/buggy.py": b"value = 1\n"},
    )
    assert (restore_dir / "pkg" / "buggy.py").read_text(encoding="utf-8") == "value = 1\n"

    blocker = restore_dir / "blocker"
    blocker.write_text("file\n", encoding="utf-8")
    patches._restore_diff_target_files(
        cwd=restore_dir,
        backups={"blocker/buggy.py": b"value = 2\n"},
    )
    assert not (restore_dir / "blocker" / "buggy.py").exists()

    assert patches._python_files_touched_by_diff(cwd=tmp_path, diff=diff) == [
        "buggy.py",
        "new_file.py",
    ]


def test_deleted_target_helpers_cover_duplicates_and_unlink_errors(
    tmp_path: Path, monkeypatch
) -> None:
    diff = (
        "diff --git a/obsolete.py b/obsolete.py\n"
        "--- a/obsolete.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-old\n"
        "diff --git a/obsolete.py b/obsolete.py\n"
        "--- a/obsolete.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-old\n"
    )
    target = tmp_path / "obsolete.py"
    target.write_text("old\n", encoding="utf-8")

    assert patches._deleted_diff_target_paths(diff) == ["obsolete.py"]

    monkeypatch.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("boom")))
    patches._remove_deleted_diff_targets(cwd=tmp_path, diff=diff)
    assert target.exists()


def test_patch_verifier_helper_functions_cover_core_matching_rules() -> None:
    diff = (
        "diff --git a/pkg/buggy.py b/pkg/buggy.py\n"
        "--- a/pkg/buggy.py\n"
        "+++ b/pkg/buggy.py\n"
        "@@ -1 +1 @@\n"
        "-return 1\n"
        "+return 2\n"
        "diff --git a/tests/test_buggy.py b/tests/test_buggy.py\n"
        "--- a/tests/test_buggy.py\n"
        "+++ b/tests/test_buggy.py\n"
        "@@ -1 +1 @@\n"
        "-assert f() == 1\n"
        "+assert f() == 2\n"
    )

    assert patches._extract_candidate_patch_files(diff) == ["pkg/buggy.py", "tests/test_buggy.py"]
    assert patches._patch_mentions_crash_file(
        patch_files=["pkg/buggy.py"],
        crash_file="other/buggy.py",
    )
    assert patches._patch_mentions_failing_test_context(
        patch_files=["tests/test_buggy.py"],
        failing_test_file="tests/test_buggy.py",
    )
    assert (
        patches._patch_mentions_crash_file(
            patch_files=["pkg/a.py", "pkg/b.py"],
            crash_file="other/c.py",
        )
        is False
    )
    assert (
        patches._patch_mentions_failing_test_context(
            patch_files=["tests/a_test.py", "tests/b_test.py"],
            failing_test_file="tests/test_buggy.py",
        )
        is False
    )
    assert patches._patch_mentions_scope_files(
        patch_files=["pkg/buggy.py"],
        scope_files={"pkg/buggy.py"},
    )
    assert patches._patch_mentions_scope_files(
        patch_files=["buggy.py"],
        scope_files={"pkg/buggy.py"},
    )
    assert (
        patches._patch_mentions_crash_file(patch_files=["pkg/buggy.py"], crash_file=None) is False
    )
    assert (
        patches._patch_mentions_failing_test_context(
            patch_files=["pkg/buggy.py"],
            failing_test_file=None,
        )
        is False
    )
    assert patches._patch_mentions_scope_files(patch_files=["buggy.py"], scope_files=set()) is False
    assert (
        patches._patch_mentions_scope_files(
            patch_files=["other/buggy.py"],
            scope_files={"pkg/buggy.py"},
        )
        is False
    )
    assert (
        patches._patch_mentions_scope_files(
            patch_files=["buggy.py"],
            scope_files={"pkg/buggy.py", "other/buggy.py"},
        )
        is False
    )
    assert patches._patch_contains_substantive_change(diff) is True
    assert (
        patches._patch_contains_substantive_change("diff --git a/a b/a\n--- a/a\n+++ b/a\n")
        is False
    )
    assert (
        patches._patch_contains_substantive_change("diff --git a/a b/a\n@@ -1 +1 @@\n keep\n\n")
        is False
    )
    assert patches._is_safe_relative_path("pkg/buggy.py") is True
    assert patches._is_safe_relative_path("../buggy.py") is False
    assert patches._is_safe_relative_path("/tmp/buggy.py") is False
    assert patches._is_safe_relative_path(".") is False
    assert patches._is_safe_relative_path("/dev/null") is False


def test_render_patch_verifier_recovery_feedback_mentions_key_fields() -> None:
    feedback = patches._render_patch_verifier_recovery_feedback(
        score=1,
        min_score=2,
        reasons=["insufficient_source_relevance"],
        patch_files=["pkg/buggy.py"],
        crash_file="pkg/buggy.py",
        inferred_source_scope=["pkg/buggy.py"],
    )

    assert "score: 1/2" in feedback
    assert "touched_files: pkg/buggy.py" in feedback
    assert "crash_file: pkg/buggy.py" in feedback


def test_forced_attempt_helpers_cover_marker_and_target_selection(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_text(
        "def buggy() -> int:\n    return 1\n# llmdebug-force-attempt-3\n",
        encoding="utf-8",
    )

    target, diff = patches._build_runner_forced_attempt_diff(
        cwd=tmp_path,
        file_index="test_buggy.py\nbuggy.py\nnotes.txt\n",
        attempt=3,
    ) or ("", "")

    assert target == "buggy.py"
    assert "# llmdebug-force-attempt-3-2" in diff
    assert patches._is_forced_noop_marker_diff(diff) is True
    assert (
        patches._is_forced_noop_marker_diff(
            "diff --git a/buggy.py b/buggy.py\n--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-a\n+b\n"
        )
        is False
    )
    assert patches._is_forced_noop_marker_diff("\n+\nplain text\n") is False
    assert patches._select_forced_patch_target(
        ["notes.txt", "tests/test_buggy.py", "buggy.py"]
    ) == ("buggy.py")
    assert (
        patches._select_forced_patch_target(["notes.txt", "tests/test_buggy.py"])
        == "tests/test_buggy.py"
    )
    assert patches._select_forced_patch_target(["notes.txt"]) is None


def test_apply_relaxed_replacements_covers_additional_edge_cases(
    tmp_path: Path, monkeypatch
) -> None:
    assert patches._apply_relaxed_replacements(tmp_path, "diff --git a/x b/x\n@@ -1 +1 @@\n") == (
        False,
        "no replaceable relaxed hunk blocks found",
    )

    outside = tmp_path.parent / "outside_relaxed"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/linked/buggy.py\n@@ -1 +1 @@\n-old\n+new\n",
    )
    assert ok is False
    assert msg == "path outside working directory: linked/buggy.py"

    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/missing.py\n@@ -1 +1 @@\n-old\n+new\n",
    )
    assert ok is False
    assert msg == "file not found: missing.py"

    target = tmp_path / "buggy.py"
    target.write_text("alpha\nalpha\n", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1 +1 @@\n-alpha\n+beta\n",
    )
    assert ok is False
    assert msg == "ambiguous relaxed replacement target block in 'buggy.py'"

    target.write_text("beta\n", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1 +1 @@\n-beta\n+beta\n",
    )
    assert ok is False
    assert msg == "relaxed replacement produced no file content changes"

    target.write_text("start\nend", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1,2 +1,3 @@\n start\n+middle\n end\n",
    )
    assert ok is True, msg
    assert target.read_text(encoding="utf-8") == "start\nmiddle\nend"


def test_apply_relaxed_replacements_rejects_old_and_insertion_context_mismatches(
    tmp_path: Path,
) -> None:
    target = tmp_path / "buggy.py"

    target.write_text("target\n", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1 +1 @@\n before\n-target\n+updated\n",
    )
    assert ok is False
    assert msg == "could not find context-anchored target block in 'buggy.py'"

    target.write_text("before\nold\nmismatch\n", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1,3 +1,3 @@\n before\n-old\n+updated\n after\n",
    )
    assert ok is False
    assert msg == "could not find context-anchored target block in 'buggy.py'"

    target.write_text("before\n", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1 +1,2 @@\n before\n+inserted\n",
    )
    assert ok is True, msg
    assert target.read_text(encoding="utf-8") == "before\ninserted\n"

    target.write_text("before\nmismatch\n", encoding="utf-8")
    ok, msg = patches._apply_relaxed_replacements(
        tmp_path,
        "+++ b/buggy.py\n@@ -1,2 +1,3 @@\n before\n+inserted\n after\n",
    )
    assert ok is False
    assert msg == "could not find context-anchored target block in 'buggy.py'"


def test_extract_relaxed_pairs_from_hunk_covers_context_and_no_newline_markers() -> None:
    pairs = patches._extract_relaxed_pairs_from_hunk(
        [
            " context0",
            " context1",
            " context2",
            " context3",
            "-old",
            "\\ No newline at end of file",
            "+new",
            " after0",
            "\\ No newline at end of file",
            " after1",
            "?ignored",
        ]
    )

    assert pairs == [(["old"], ["new"], ["context2", "context3"], ["after0", "after1"])]


def test_build_runner_forced_attempt_diff_covers_missing_and_blank_diff_cases(
    tmp_path: Path, monkeypatch
) -> None:
    assert patches._build_runner_forced_attempt_diff(cwd=tmp_path, file_index="", attempt=1) is None
    assert (
        patches._build_runner_forced_attempt_diff(
            cwd=tmp_path,
            file_index="README.md\nnotes.txt\n",
            attempt=1,
        )
        is None
    )
    assert (
        patches._build_runner_forced_attempt_diff(
            cwd=tmp_path,
            file_index="buggy.py\n",
            attempt=1,
        )
        is None
    )

    outside = tmp_path.parent / "outside_force"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    (outside / "buggy.py").write_text("x = 1\n", encoding="utf-8")
    assert (
        patches._build_runner_forced_attempt_diff(
            cwd=tmp_path,
            file_index="linked/buggy.py\n",
            attempt=1,
        )
        is None
    )

    real_target = tmp_path / "real.py"
    real_target.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(patches, "_build_diff_from_changes", lambda changes: "   ")
    assert (
        patches._build_runner_forced_attempt_diff(
            cwd=tmp_path,
            file_index="real.py\n",
            attempt=1,
        )
        is None
    )


def test_build_diff_from_changes_returns_empty_string_for_noops() -> None:
    assert patches._build_diff_from_changes([("buggy.py", "same\n", "same\n")]) == ""
