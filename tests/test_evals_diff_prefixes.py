"""git's a/ and b/ prefixes are stripped only as a header pair (LLM-23)."""

from __future__ import annotations

from pathlib import Path

from evals.eval_patches import apply_diff
from evals.patchers.openai_diff_text import (
    _extract_unified_diff,
    _fix_hunk_counts,
    _strip_git_prefixes,
)


def test_git_style_pairs_lose_their_prefixes() -> None:
    assert _strip_git_prefixes("a/pkg/mod.py", "b/pkg/mod.py") == ("pkg/mod.py", "pkg/mod.py")
    assert _strip_git_prefixes("/dev/null", "b/new.py") == ("/dev/null", "new.py")
    assert _strip_git_prefixes("a/old.py", "/dev/null") == ("old.py", "/dev/null")


def test_a_real_top_level_directory_named_a_survives() -> None:
    assert _strip_git_prefixes("a/mod.py", "a/mod.py") == ("a/mod.py", "a/mod.py")
    assert _strip_git_prefixes("b/mod.py", "b/mod.py") == ("b/mod.py", "b/mod.py")


def test_prefix_less_headers_for_directory_a_reach_the_right_file() -> None:
    diff = "--- a/mod.py\n+++ a/mod.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    normalized = _extract_unified_diff(diff)
    assert normalized is not None
    assert "--- a/a/mod.py" in normalized
    assert "+++ b/a/mod.py" in normalized


def test_creation_into_a_real_directory_named_b_lands_there(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    response = "diff --git b/mod.py b/mod.py\n--- /dev/null\n+++ b/mod.py\n@@ -0,0 +1 @@\n+x = 1\n"

    diff = _extract_unified_diff(response)
    assert diff is not None
    ok, message = apply_diff(tmp_path, _fix_hunk_counts(diff))

    assert ok, message
    assert (tmp_path / "b" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (tmp_path / "mod.py").exists()


def test_deletion_in_a_real_directory_named_a_spares_the_root_file(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    response = "diff --git a/mod.py a/mod.py\n--- a/mod.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"

    diff = _extract_unified_diff(response)
    assert diff is not None
    ok, message = apply_diff(tmp_path, _fix_hunk_counts(diff))

    assert ok, message
    assert not (tmp_path / "a" / "mod.py").exists()
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


def test_git_header_resolves_the_dev_null_side_prefix() -> None:
    # "/dev/null" is no evidence of a prefix pair; the git header decides (LLM-23).
    assert _strip_git_prefixes("/dev/null", "b/mod.py", declared="b/mod.py") == (
        "/dev/null",
        "b/mod.py",
    )
    assert _strip_git_prefixes("/dev/null", "b/mod.py", declared="mod.py") == (
        "/dev/null",
        "mod.py",
    )
    assert _strip_git_prefixes("a/mod.py", "/dev/null", declared="a/mod.py") == (
        "a/mod.py",
        "/dev/null",
    )
