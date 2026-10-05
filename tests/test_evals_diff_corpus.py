"""End-to-end diff corpus (2026-10-04 review, Phase 1G).

Realistic model outputs go through the real pipeline used by the OpenAI-compatible
patcher in diff mode, ``_extract_unified_diff`` -> ``_fix_hunk_counts`` -> ``apply_diff``,
and the test asserts the files on disk afterwards. Each case names the finding it covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.eval_patches import _crlf_diff_targets, apply_diff
from evals.patchers import openai_compat_patcher as openai_mod
from evals.patchers.openai_compat_patcher import _extract_unified_diff, _fix_hunk_counts

A = "def f():\n    return 1\n"
C = "def f():\n    return 1\n\n\ndef h():\n    return 3\n"
B = "def g():\n    return 10\n"
A_FIXED = "def f():\n    return 2\n"
B_FIXED = "def g():\n    return 20\n"

DIFF_A = (
    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
)
DIFF_B = (
    "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n"
    "@@ -1,2 +1,2 @@\n def g():\n-    return 10\n+    return 20\n"
)
PLAIN_A = "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"


def _workdir(tmp_path: Path, *, crlf: bool = False) -> Path:
    for name, text in (("a.py", A), ("b.py", B), ("c.py", C)):
        (tmp_path / name).write_bytes(text.replace("\n", "\r\n" if crlf else "\n").encode())
    return tmp_path


def _pipeline(tmp_path: Path, response: str) -> tuple[bool, str]:
    diff = _extract_unified_diff(response)
    assert diff is not None, "no diff extracted"
    return apply_diff(tmp_path, _fix_hunk_counts(diff))


def _read(path: Path) -> str:
    return path.read_bytes().decode()


CASES: dict[str, tuple[str, dict[str, str]]] = {
    "one_fence": (f"Here is the fix:\n```diff\n{DIFF_A}```\n", {"a.py": A_FIXED}),
    "adjacent_diffs_one_fence": (
        f"```diff\n{DIFF_A}{DIFF_B}```\n",
        {"a.py": A_FIXED, "b.py": B_FIXED},
    ),
    "two_fences": (  # LLM-01
        f"```diff\n{DIFF_A}```\nAlso:\n```diff\n{DIFF_B}```\n",
        {"a.py": A_FIXED, "b.py": B_FIXED},
    ),
    "prose_between_diffs_one_fence": (  # LLM-01
        f"```diff\n{DIFF_A}\nAnd the second file:\n{DIFF_B}```\n",
        {"a.py": A_FIXED, "b.py": B_FIXED},
    ),
    "unfenced_prose_between": (  # LLM-01
        f"{DIFF_A}\nNow b.py:\n{DIFF_B}",
        {"a.py": A_FIXED, "b.py": B_FIXED},
    ),
    "trailing_markdown_bullets": (  # LLM-07
        f"{DIFF_A}- This changes the return value to 2.\n- No other changes.\n",
        {"a.py": A_FIXED},
    ),
    "blank_context_line_without_space": (  # LLM-03
        "--- a/c.py\n+++ b/c.py\n@@ -1,5 +1,5 @@\n def f():\n-    return 1\n+    return 2\n"
        "\n\n def h():\n",
        {"c.py": "def f():\n    return 2\n\n\ndef h():\n    return 3\n"},
    ),
    "change_after_consecutive_blank_context_lines": (  # LLM-03
        "--- a/c.py\n+++ b/c.py\n@@ -1,6 +1,6 @@\n def f():\n     return 1\n"
        "\n\n def h():\n-    return 3\n+    return 4\n",
        {"c.py": "def f():\n    return 1\n\n\ndef h():\n    return 4\n"},
    ),
    "orphan_think_close_with_draft": (  # LLM-04
        "Let me think. Draft:\n" + DIFF_A.replace("return 2", "return 99") + "</think>\n"
        f"```diff\n{DIFF_A}```\n",
        {"a.py": A_FIXED},
    ),
    "matched_think_block": (
        f"<think>maybe {DIFF_A.replace('return 2', 'return 7')}</think>\n{DIFF_A}",
        {"a.py": A_FIXED},
    ),
    "unclosed_think_after_answer": (  # LLM-04
        f"{DIFF_A}<think>\nhmm, maybe also change b.py:\n{DIFF_B}",
        {"a.py": A_FIXED, "b.py": B},
    ),
    "draft_then_final_same_file": (  # LLM-04: prefer the last diff for a file
        f"First try:\n```diff\n{DIFF_A.replace('return 2', 'return 5')}```\n"
        f"Actually, better:\n```diff\n{DIFF_A}```\n",
        {"a.py": A_FIXED},
    ),
    "new_file_via_plain_headers": (  # LLM-10
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+def n():\n+    return 0\n",
        {"new.py": "def n():\n    return 0\n", "a.py": A},
    ),
    "plain_unified_headers": (f"```\n{PLAIN_A}```", {"a.py": A_FIXED}),
    "miscounted_hunk_header": (
        "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,5 @@\n def f():\n-    return 1\n+    return 2\n",
        {"a.py": A_FIXED},
    ),
    "form_feed_inside_a_line": (  # LLM-15
        "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2  # \x0c\n",
        {"a.py": "def f():\n    return 2  # \x0c\n"},
    ),
    "git_diff_with_index_line": (
        DIFF_A.replace("--- a/a.py", "index 83db48f..bf269f4 100644\n--- a/a.py"),
        {"a.py": A_FIXED},
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_diff_corpus(tmp_path: Path, name: str) -> None:
    response, expected = CASES[name]
    root = _workdir(tmp_path)
    ok, message = _pipeline(root, response)
    assert ok, message
    for filename, content in expected.items():
        assert _read(root / filename) == content, filename


def test_crlf_file_keeps_its_line_endings(tmp_path: Path) -> None:  # LLM-09 (diff path)
    root = _workdir(tmp_path, crlf=True)
    ok, message = _pipeline(root, DIFF_A)
    assert ok, message
    assert _read(root / "a.py") == A_FIXED.replace("\n", "\r\n")


def test_failed_apply_leaves_crlf_file_byte_identical(tmp_path: Path) -> None:  # LLM-09
    root = _workdir(tmp_path, crlf=True)
    before = (root / "a.py").read_bytes()
    ok, _ = apply_diff(root, DIFF_A.replace(" def f():", " def missing():"))
    assert not ok
    assert (root / "a.py").read_bytes() == before


def test_only_pure_crlf_files_are_normalized(tmp_path: Path) -> None:  # LLM-09
    (tmp_path / "a.py").write_bytes(b"def f():\r\n    return 1\n")  # mixed: left to `patch`
    (tmp_path / "b.py").write_bytes(B.replace("\n", "\r\n").encode())
    assert _crlf_diff_targets(tmp_path, DIFF_A + DIFF_B) == [tmp_path / "b.py"]


def test_truncated_response_is_not_applied(  # LLM-02
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truncated = DIFF_A.rsplit("+", 1)[0]  # cut off before the replacement line
    root = _workdir(tmp_path)
    # Without the finish_reason check, the pipeline turns this into a valid deletion.
    ok, _ = _pipeline(root, truncated)
    assert ok
    assert _read(root / "a.py") == "def f():\n"

    response = {"choices": [{"message": {"content": truncated}, "finish_reason": "length"}]}
    monkeypatch.setattr(
        openai_mod,
        "_chat_completion",
        lambda **_: (response, {"finish_reason": "length", "llm_call_count": 1}),
    )
    result = openai_mod._propose_patch_via_diff(
        url="http://127.0.0.1:1/v1/chat/completions",
        api_key="dummy",
        model="dummy",
        prompt="fix it",
        temperature=0.0,
        max_tokens=16,
        timeout_sec=5.0,
        seed=None,
        http2=False,
    )
    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "truncated_response"
