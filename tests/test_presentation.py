"""Focused tests for plain-text presentation helpers."""

from __future__ import annotations

from llmdebug._presentation import format_frame_text


def test_format_frame_text_skips_internal_py_locals() -> None:
    frame = {
        "file_rel": "buggy.py",
        "line": 12,
        "function": "run",
        "locals": {
            "@py_assert1": "hidden",
            "visible": 3,
        },
        "locals_meta": "not-a-dict",
    }

    rendered = format_frame_text(frame, index=0, is_crash=False)

    assert "visible = 3" in rendered
    assert "@py_assert1" not in rendered
