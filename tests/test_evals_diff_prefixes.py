"""git's a/ and b/ prefixes are stripped only as a header pair (LLM-23)."""

from __future__ import annotations

from evals.patchers.openai_diff_text import _extract_unified_diff, _strip_git_prefixes


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
