"""Tests for RCA identity/signature helpers."""

from __future__ import annotations

import string
from pathlib import Path

import pytest

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


def _snapshot_with_message(message: str) -> dict[str, object]:
    return {
        "name": "test_parse",
        "exception": {"type": "ValueError", "message": message},
        "frames": [{"file_rel": "app.py", "line": 42, "function": "parse"}],
        "crash_frame_index": 0,
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("<Item object at 0x7f3a2c1b9d30> is stale", "<Item object at 0x10a4f2e80> is stale"),
        (
            "missing /tmp/pytest-of-ci/pytest-12/test_parse0/data.json",
            "missing /tmp/pytest-of-ci/pytest-13/test_parse0/data.json",
        ),
        ("expected 3 rows, got 4", "expected 3 rows, got 5"),
        (
            "job 2f1c7a9e-3b4d-4e5f-8a6b-1c2d3e4f5a6b failed",
            "job 9a8b7c6d-1e2f-4a3b-9c4d-5e6f7a8b9c0d failed",
        ),
        ("cache key d41d8cd98f00b204 missing", "cache key 9e107d9d372bb682 missing"),
        (
            "C:\\Users\\ci\\AppData\\Local\\Temp\\a1\\x.txt gone",
            "C:\\Users\\ci\\AppData\\Local\\Temp\\b7\\x.txt gone",
        ),
    ],
)
def test_signature_ignores_run_specific_values(first: str, second: str) -> None:
    """RCA-12: the same failure with a new address, tmp path, id or count keeps its signature."""
    assert build_failure_signature(_snapshot_with_message(first)) == build_failure_signature(
        _snapshot_with_message(second)
    )
    # The raw signature (normalize=False) is what the eval harness pins.
    assert build_failure_signature(
        _snapshot_with_message(first), normalize=False
    ) != build_failure_signature(_snapshot_with_message(second), normalize=False)


def test_signature_still_separates_different_failures() -> None:
    assert build_failure_signature(
        _snapshot_with_message("invalid literal for int()")
    ) != build_failure_signature(_snapshot_with_message("could not convert string to float"))


def test_text_signature_ignores_timings() -> None:
    first = build_failure_signature(None, stdout="FAILED test_a\n1 failed in 0.23s")
    second = build_failure_signature(None, stdout="FAILED test_a\n1 failed in 0.31s")

    assert first == second


def test_normalize_failure_text_placeholders() -> None:
    from llmdebug.rca_identity import normalize_failure_text

    assert normalize_failure_text("at 0xdeadBEEF in /srv/app/x.py line 7, ratio 3/4") == (
        "at <addr> in <path> line <n>, ratio <n>/<n>"
    )


def test_stall_fires_when_only_the_tmp_path_changes(tmp_path: Path) -> None:
    """RCA-12 end to end: MCP-style advances from snapshots, as `llmdebug_rca_advance` does."""
    from llmdebug.mcp_rca import advance_rca
    from llmdebug.rca_types import RCAAttemptRequest, RCASnapshotAdvanceRequest

    record = None
    # The first advance binds new evidence; the next three repeat it without progress.
    for run in range(4):
        snapshot = _snapshot_with_message(f"missing /tmp/pytest-{run}/test_parse0/data.json")
        record = advance_rca(
            request=RCASnapshotAdvanceRequest(
                out_dir=str(tmp_path),
                snapshot=snapshot,
                attempt=RCAAttemptRequest(
                    session_id="",
                    action="bind_evidence",
                    failure_signature="",
                    evidence_refs=("crash_frame:app.py:42:parse",),
                    gate_mode="soft",
                ),
            )
        )

    assert record is not None
    assert record["repeat_count"] == 4
    assert record["repair_streak"] == 3
    assert record["state"] == "stalled"
