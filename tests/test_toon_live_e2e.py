"""Live end-to-end integration tests for TOON encoding in adaptive conditions.

These tests verify that TOON encoding produces correct behavior when wired
into a real eval flow.  They do NOT require a running LLM server; instead
they test the encoding/decoding pipeline with realistic snapshot data.

Run with:
    uv run pytest tests/test_toon_live_e2e.py -v

The @integration-marked tests (if any) require a running LLM server.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from evals.toon_llm_encoder import (
    decode_for_llm,
    encode_for_llm,
    estimate_token_savings,
)


def _toons_installed() -> bool:
    try:
        import toons  # type: ignore[import-untyped]

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Realistic snapshot fixture (mirrors a real llmdebug capture)
# ---------------------------------------------------------------------------


@pytest.fixture
def realistic_snapshot() -> dict[str, Any]:
    """A realistic multi-frame snapshot similar to what the eval harness sees."""
    return {
        "schema_version": 1,
        "timestamp_utc": "2024-06-15T10:30:00Z",
        "llmdebug_version": "0.9.0",
        "exception": {
            "type": "AssertionError",
            "qualified_type": "builtins.AssertionError",
            "message": "assert [1, 2, 3] == [1, 2, 4]",
            "error_category": {
                "category": "assertion_failure",
                "suggestion": "Check the expected values in the assertion.",
            },
        },
        "frames": [
            {
                "file": "/home/user/project/tests/test_algo.py",
                "file_rel": "tests/test_algo.py",
                "line": 42,
                "function": "test_sort",
                "module": "tests.test_algo",
                "code": "assert sort_list(data) == expected",
                "locals": {
                    "data": [3, 1, 2],
                    "expected": [1, 2, 4],
                    "result": [1, 2, 3],
                    "self": "<TestSort instance>",
                },
                "source": {
                    "start": 38,
                    "end": 46,
                    "snippet": "def test_sort(self):\n    data = [3, 1, 2]\n    expected = [1, 2, 4]\n    assert sort_list(data) == expected\n",
                },
            },
            {
                "file": "/home/user/project/src/algo.py",
                "file_rel": "src/algo.py",
                "line": 15,
                "function": "sort_list",
                "module": "algo",
                "code": "return sorted(items)",
                "locals": {"items": [3, 1, 2]},
                "source": {
                    "start": 12,
                    "end": 18,
                    "snippet": "def sort_list(items):\n    return sorted(items)\n",
                },
            },
            {
                "file": "/home/user/project/src/utils.py",
                "file_rel": "src/utils.py",
                "line": 5,
                "function": "validate",
                "module": "utils",
                "code": "if not items: raise ValueError",
                "locals": {"items": [3, 1, 2]},
            },
        ],
        "git": {
            "commit": "a1b2c3d",
            "commit_full": "a1b2c3d4e5f6g7h8i9j0",
            "branch": "main",
            "dirty": True,
        },
        "env": {
            "python": "3.11.5",
            "platform": "Linux-5.15.0",
            "cwd": "/home/user/project",
        },
        "pytest": {
            "nodeid": "tests/test_algo.py::TestSort::test_sort",
            "outcome": "failed",
        },
    }


# ---------------------------------------------------------------------------
# TOON vs JSON token comparison
# ---------------------------------------------------------------------------


class TestToonVsJsonComparison:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_toon_shorter_than_pretty_json(self, realistic_snapshot: dict[str, Any]) -> None:
        """TOON encoding should be shorter than pretty-printed JSON."""
        toon_encoded = encode_for_llm(realistic_snapshot, use_toon=True, use_compact_keys=True)
        json_pretty = json.dumps(realistic_snapshot, indent=2, ensure_ascii=False)
        assert len(toon_encoded) < len(json_pretty)

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_token_savings_at_least_15_percent(self, realistic_snapshot: dict[str, Any]) -> None:
        """Realistic snapshots should yield >= 15% savings with TOON + compact keys."""
        savings = estimate_token_savings(realistic_snapshot, use_toon=True, use_compact_keys=True)
        assert savings["savings_frac"] >= 0.15, (
            f"Expected >= 15% savings, got {savings['savings_frac']:.1%}"
        )

    def test_compact_keys_only_savings(self, realistic_snapshot: dict[str, Any]) -> None:
        """Even without TOON, compact keys + minified JSON should save tokens."""
        savings = estimate_token_savings(realistic_snapshot, use_toon=False, use_compact_keys=True)
        assert savings["savings_frac"] > 0.0


# ---------------------------------------------------------------------------
# Full round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTripFidelity:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_full_roundtrip_all_fields(self, realistic_snapshot: dict[str, Any]) -> None:
        """encode → decode should preserve all snapshot fields."""
        encoded = encode_for_llm(realistic_snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        assert decoded["schema_version"] == realistic_snapshot["schema_version"]
        assert decoded["timestamp_utc"] == realistic_snapshot["timestamp_utc"]
        assert decoded["exception"]["type"] == "AssertionError"
        assert decoded["exception"]["message"] == realistic_snapshot["exception"]["message"]
        assert len(decoded["frames"]) == len(realistic_snapshot["frames"])

        # Check frame-level fidelity
        for i, (dec_frame, orig_frame) in enumerate(
            zip(decoded["frames"], realistic_snapshot["frames"], strict=True)
        ):
            assert dec_frame["file"] == orig_frame["file"], f"Frame {i} file mismatch"
            assert dec_frame["line"] == orig_frame["line"], f"Frame {i} line mismatch"
            assert dec_frame["function"] == orig_frame["function"], f"Frame {i} function mismatch"
            assert dec_frame["locals"] == orig_frame["locals"], f"Frame {i} locals mismatch"

        # Git context preserved
        assert decoded["git"]["commit"] == "a1b2c3d"
        assert decoded["git"]["dirty"] is True

    def test_full_roundtrip_json_mode(self, realistic_snapshot: dict[str, Any]) -> None:
        """JSON fallback round-trip also preserves all fields."""
        encoded = encode_for_llm(realistic_snapshot, use_toon=False, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        assert decoded["schema_version"] == realistic_snapshot["schema_version"]
        assert decoded["exception"]["type"] == "AssertionError"
        assert len(decoded["frames"]) == 3
        assert decoded["frames"][0]["locals"]["data"] == [3, 1, 2]


# ---------------------------------------------------------------------------
# TOON encoding does not change solver-relevant semantics
# ---------------------------------------------------------------------------


class TestSolverSemanticParity:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_exception_info_identical(self, realistic_snapshot: dict[str, Any]) -> None:
        """The exception info visible to the solver should be identical."""
        encoded = encode_for_llm(realistic_snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        assert decoded["exception"] == realistic_snapshot["exception"]

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_frame_code_identical(self, realistic_snapshot: dict[str, Any]) -> None:
        """The code lines in frames (critical for repair) should be identical."""
        encoded = encode_for_llm(realistic_snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        for i, (dec, orig) in enumerate(
            zip(decoded["frames"], realistic_snapshot["frames"], strict=True)
        ):
            assert dec.get("code") == orig.get("code"), f"Frame {i} code mismatch"

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_locals_values_identical(self, realistic_snapshot: dict[str, Any]) -> None:
        """Local variable values must be perfectly preserved."""
        encoded = encode_for_llm(realistic_snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)

        for i, (dec, orig) in enumerate(
            zip(decoded["frames"], realistic_snapshot["frames"], strict=True)
        ):
            assert dec.get("locals") == orig.get("locals"), f"Frame {i} locals mismatch"
