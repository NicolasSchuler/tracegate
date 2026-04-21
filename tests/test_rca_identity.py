"""Tests for RCA identity/signature helpers."""

from __future__ import annotations

import string
from pathlib import Path

from llmdebug.rca_identity import build_failure_signature, derive_session_id


def test_derive_session_id_prefers_pytest_nodeid(tmp_path: Path) -> None:
    snapshot = {
        "name": "fallback_name",
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
        "exception": {"type": "ValueError", "message": "boom"},
        "frames": [],
        "crash_frame_index": 0,
    }

    session_id = derive_session_id(snapshot, out_dir=str(tmp_path), fallback="mcp")
    assert session_id.endswith("::tests/test_app.py::test_parse")


def test_derive_session_id_falls_back_to_snapshot_name(tmp_path: Path) -> None:
    snapshot = {
        "name": "test_case_alpha",
        "exception": {"type": "RuntimeError", "message": "boom"},
        "frames": [],
        "crash_frame_index": 0,
    }

    session_id = derive_session_id(snapshot, out_dir=str(tmp_path), fallback="mcp")
    assert session_id.endswith("::test_case_alpha")


def test_build_failure_signature_text_fallback_is_stable() -> None:
    stderr = "Traceback\nValueError: bad input"
    sig_a = build_failure_signature(None, stdout="", stderr=stderr)
    sig_b = build_failure_signature(None, stdout="", stderr=stderr)

    assert sig_a == sig_b
    assert sig_a.startswith("TextFailure:")


def test_build_failure_signature_text_fallback_uses_fixed_hex_digest() -> None:
    signature = build_failure_signature(None, stdout="", stderr="ValueError: bad input")
    digest = signature.split(":", 1)[1]

    assert len(digest) == 16
    assert set(digest) <= set(string.hexdigits.lower())


def test_build_failure_signature_handles_non_numeric_crash_line(tmp_path: Path) -> None:
    snapshot = {
        "exception": {"type": "ValueError", "message": "boom"},
        "frames": [{"file": "/code/app.py", "line": "?", "function": "parse_input"}],
        "crash_frame_index": 0,
    }

    signature = build_failure_signature(snapshot, stdout="", stderr="")

    assert signature.startswith("ValueError:")


def test_derive_session_id_uses_fixed_hex_path_prefix(tmp_path: Path) -> None:
    session_id = derive_session_id(None, out_dir=str(tmp_path), fallback="mcp")
    path_hash, scope = session_id.split("::", 1)

    assert len(path_hash) == 8
    assert set(path_hash) <= set(string.hexdigits.lower())
    assert scope == "mcp"


def test_build_failure_signature_handles_non_dict_frame() -> None:
    """Branch [27, 29]: frame candidate is not a dict — crash_frame stays empty."""
    snapshot = {
        "exception": {"type": "ValueError", "message": "boom"},
        "frames": ["not_a_dict"],
        "crash_frame_index": 0,
    }
    sig = build_failure_signature(snapshot)
    assert sig.startswith("ValueError:")


def test_derive_session_id_with_empty_nodeid_falls_back_to_name(tmp_path: Path) -> None:
    """Branch [89, 91]: nodeid present but falsy — falls through to name check."""
    snapshot = {
        "name": "my_test",
        "pytest": {"nodeid": ""},
        "exception": {"type": "ValueError"},
        "frames": [],
    }
    session_id = derive_session_id(snapshot, out_dir=str(tmp_path), fallback="global")
    assert session_id.endswith("::my_test")


def test_derive_session_id_with_empty_name_uses_fallback(tmp_path: Path) -> None:
    """Branch [93, 95]: name present but falsy — uses fallback."""
    snapshot = {
        "name": "",
        "exception": {"type": "ValueError"},
        "frames": [],
    }
    session_id = derive_session_id(snapshot, out_dir=str(tmp_path), fallback="global")
    assert session_id.endswith("::global")
