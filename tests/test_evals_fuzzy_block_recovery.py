"""Fuzzy block recovery stays cheap and decides as before (LLM-12)."""

from __future__ import annotations

import difflib
import random
import time

from evals.patchers.openai_edits_apply import (
    _apply_fuzzy_block_replacement,
    _compile_whitespace_normalized_pattern,
)


def _exhaustive(before: str, find: str, replace: str) -> str | None:
    """The pre-LLM-12 search: score every window with ratio()."""
    find_block = find.strip()
    lines = before.splitlines(keepends=True)
    target = len(find_block.splitlines())
    sizes = {target} | ({target - 1, target + 1} if target > 1 else set())
    sizes = {n for n in sizes if 1 <= n <= len(lines)}
    offsets = [0]
    for chunk in lines:
        offsets.append(offsets[-1] + len(chunk))
    scored = []
    for size in sorted(sizes):
        for start in range(len(lines) - size + 1):
            window = "".join(lines[start : start + size]).strip()
            if window:
                ratio = difflib.SequenceMatcher(None, window, find_block).ratio()
                scored.append((ratio, start, size))
    scored.sort(key=lambda item: item[0], reverse=True)
    best, start, size = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best < (0.96 if target == 1 else 0.92) or best - second < 0.03:
        return None
    return before[: offsets[start]] + replace + before[offsets[start + size] :]


def test_pruned_search_decides_like_the_exhaustive_one() -> None:
    rng = random.Random(7)
    words = ["alpha", "beta", "gamma", "delta", "value", "result", "items", "count"]
    for _ in range(40):
        lines = [
            f"    {rng.choice(words)} = {rng.choice(words)}({rng.randint(0, 9)})\n"
            for _ in range(60)
        ]
        before = "".join(lines)
        start = rng.randint(0, 55)
        find = "".join(lines[start : start + 3])
        if rng.random() < 0.5:
            find = find.replace("(", " (", 1)  # a near miss
        expected = _exhaustive(before, find, "REPLACED\n")
        actual = _apply_fuzzy_block_replacement(
            before_nl=before, find_nl=find, replace_nl="REPLACED\n"
        )
        assert actual == expected


def test_a_failed_search_on_a_large_file_is_fast() -> None:
    rng = random.Random(1)
    before = "".join(f"line_{rng.randint(0, 10**9)} = {i}\n" for i in range(5900))
    find = "".join(f"absent_{i} = compute({i})\n" for i in range(12))
    started = time.perf_counter()
    assert _apply_fuzzy_block_replacement(before_nl=before, find_nl=find, replace_nl="x\n") is None
    assert time.perf_counter() - started < 3.0


def test_whitespace_pattern_ignores_trailing_blanks_without_backtracking() -> None:
    pattern = _compile_whitespace_normalized_pattern("x = 1 \ny = 2")
    assert pattern.search("x = 1\ny = 2")
    assert pattern.search("x  =  1 \t\ny = 2")

    nasty = _compile_whitespace_normalized_pattern("a" + " " * 40 + "\nb")
    started = time.perf_counter()
    assert nasty.search("a" + " " * 4000 + "c") is None
    assert time.perf_counter() - started < 0.5


def _transposed(text: str, *, every: int) -> str:
    """*text* with adjacent characters swapped, which keeps quick_ratio at 1.0."""
    chars = list(text)
    for index in range(0, len(chars) - 1, every):
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


def test_one_huge_line_cannot_outlast_the_budget_inside_a_single_comparison() -> None:
    # Many distinct characters defeat difflib's autojunk, so ratio() does the full
    # quadratic work; the cheap bounds both return 1.0 for this pair.
    rng = random.Random(7)
    alphabet = [chr(0x4E00 + offset) for offset in range(400)]
    line = "".join(rng.choice(alphabet) for _ in range(16_000))
    find = _transposed(line, every=97)
    matcher = difflib.SequenceMatcher(None, line, find)
    assert matcher.real_quick_ratio() == 1.0
    assert matcher.quick_ratio() == 1.0

    started = time.perf_counter()
    result = _apply_fuzzy_block_replacement(
        before_nl=f"{line}\n", find_nl=find, replace_nl="x = 1\n"
    )
    elapsed = time.perf_counter() - started

    assert result is None
    assert elapsed < 2.0  # one ratio() call on this pair alone took seconds
