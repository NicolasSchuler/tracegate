"""Tests for evals.toon_llm_encoder module."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals.toon_llm_encoder import (
    _toons_available,
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_snapshot() -> dict[str, Any]:
    """A minimal but realistic snapshot with frames (uniform array)."""
    return {
        "schema_version": 1,
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "exception": {
            "type": "ValueError",
            "message": "invalid literal for int()",
        },
        "frames": [
            {
                "file": "main.py",
                "file_rel": "main.py",
                "line": 42,
                "function": "run",
                "module": "__main__",
                "code": "x = int(val)",
                "locals": {"val": "abc", "x": None},
            },
            {
                "file": "lib.py",
                "file_rel": "lib.py",
                "line": 10,
                "function": "parse",
                "module": "lib",
                "code": "return int(s)",
                "locals": {"s": "abc"},
            },
        ],
    }


@pytest.fixture
def empty_snapshot() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "exception": {"type": "RuntimeError", "message": "empty"},
        "frames": [],
    }


@pytest.fixture
def nested_locals_snapshot() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "exception": {"type": "TypeError", "message": "oops"},
        "frames": [
            {
                "file": "deep.py",
                "line": 1,
                "function": "f",
                "locals": {
                    "config": {"nested": {"level": 3, "items": [1, 2, 3]}},
                    "data": [{"a": 1}, {"a": 2}],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# encode_for_llm: JSON fallback (use_toon=False)
# ---------------------------------------------------------------------------


class TestEncodeForLlmJsonFallback:
    def test_returns_minified_json(self, sample_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=False)
        parsed = json.loads(result)
        assert parsed["schema_version"] == 1
        # Minified = no indent whitespace
        assert "\n" not in result

    def test_applies_compact_keys_by_default(self, sample_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=True)
        parsed = json.loads(result)
        # Compact key for "schema_version" is "_v"
        assert "_v" in parsed
        assert "schema_version" not in parsed

    def test_no_compact_keys_when_disabled(self, sample_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=False)
        parsed = json.loads(result)
        assert "schema_version" in parsed

    def test_preserves_user_data_keys(self, sample_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=True)
        parsed = json.loads(result)
        frames = parsed.get("_fr", [])
        assert len(frames) > 0
        # User-data key "val" in locals should NOT be compacted
        locals_dict = frames[0].get("_loc", {})
        assert "val" in locals_dict

    def test_empty_snapshot(self, empty_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(empty_snapshot, use_toon=False, use_compact_keys=True)
        parsed = json.loads(result)
        assert parsed.get("_fr") == []

    def test_nested_locals(self, nested_locals_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(nested_locals_snapshot, use_toon=False, use_compact_keys=True)
        parsed = json.loads(result)
        frames = parsed.get("_fr", [])
        loc = frames[0].get("_loc", {})
        # Nested user data should be preserved as-is
        assert loc["config"]["nested"]["level"] == 3


# ---------------------------------------------------------------------------
# encode_for_llm: TOON mode
# ---------------------------------------------------------------------------


class TestEncodeForLlmToon:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_toon_with_compact_keys(self, sample_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(sample_snapshot, use_toon=True, use_compact_keys=True)
        assert isinstance(result, str)
        assert len(result) > 0
        # TOON output should be shorter than pretty JSON
        json_baseline = json.dumps(sample_snapshot, indent=2)
        assert len(result) < len(json_baseline)

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_toon_without_compact_keys(self, sample_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(sample_snapshot, use_toon=True, use_compact_keys=False)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_toon_empty_snapshot(self, empty_snapshot: dict[str, Any]) -> None:
        result = encode_for_llm(empty_snapshot, use_toon=True, use_compact_keys=True)
        assert isinstance(result, str)

    def test_graceful_fallback_when_toons_missing(
        self, monkeypatch: pytest.MonkeyPatch, sample_snapshot: dict[str, Any]
    ) -> None:
        """When toons is not importable, falls back to compact JSON."""
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        # Reset the warning flag
        import evals.toon_llm_encoder as mod

        monkeypatch.setattr(mod, "_TOON_FALLBACK_WARNING_EMITTED", False)

        with pytest.warns(match="toons package not installed"):
            result = encode_for_llm(sample_snapshot, use_toon=True, use_compact_keys=True)

        # Should be valid JSON (the fallback path)
        parsed = json.loads(result)
        assert "_v" in parsed  # compact keys applied

    def test_fallback_warning_emitted_once(
        self, monkeypatch: pytest.MonkeyPatch, sample_snapshot: dict[str, Any]
    ) -> None:
        """The fallback warning should only fire once per process."""
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        import evals.toon_llm_encoder as mod

        monkeypatch.setattr(mod, "_TOON_FALLBACK_WARNING_EMITTED", False)

        with pytest.warns(match="toons"):
            encode_for_llm(sample_snapshot, use_toon=True)

        # Second call should NOT warn
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            encode_for_llm(sample_snapshot, use_toon=True)


# ---------------------------------------------------------------------------
# decode_for_llm
# ---------------------------------------------------------------------------


class TestDecodeForLlm:
    def test_decode_json_with_expand(self, sample_snapshot: dict[str, Any]) -> None:
        encoded = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)
        assert "schema_version" in decoded
        assert decoded["schema_version"] == 1

    def test_decode_json_without_expand(self, sample_snapshot: dict[str, Any]) -> None:
        encoded = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=False)
        assert "_v" in decoded

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_roundtrip_toon(self, sample_snapshot: dict[str, Any]) -> None:
        encoded = encode_for_llm(sample_snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)
        assert decoded["schema_version"] == 1
        assert decoded["exception"]["type"] == "ValueError"
        assert len(decoded["frames"]) == 2

    def test_roundtrip_json(self, sample_snapshot: dict[str, Any]) -> None:
        encoded = encode_for_llm(sample_snapshot, use_toon=False, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)
        assert decoded["schema_version"] == 1
        assert decoded["exception"]["type"] == "ValueError"
        assert len(decoded["frames"]) == 2
        # User locals preserved
        assert decoded["frames"][0]["locals"]["val"] == "abc"

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_roundtrip_preserves_nested_locals(
        self, nested_locals_snapshot: dict[str, Any]
    ) -> None:
        encoded = encode_for_llm(nested_locals_snapshot, use_toon=True, use_compact_keys=True)
        decoded = decode_for_llm(encoded, expand=True)
        loc = decoded["frames"][0]["locals"]
        assert loc["config"]["nested"]["level"] == 3
        assert loc["data"] == [{"a": 1}, {"a": 2}]

    def test_decode_raises_on_non_dict(self) -> None:
        with pytest.raises(ValueError, match="Could not decode"):
            decode_for_llm("[1,2,3]")


# ---------------------------------------------------------------------------
# estimate_token_savings
# ---------------------------------------------------------------------------


class TestEstimateTokenSavings:
    def test_savings_positive_with_compact_keys(self, sample_snapshot: dict[str, Any]) -> None:
        result = estimate_token_savings(sample_snapshot, use_toon=False, use_compact_keys=True)
        assert result["savings_frac"] > 0.0
        assert result["savings_chars"] > 0
        assert result["encoding"] == "json_compact"

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_savings_positive_with_toon(self, sample_snapshot: dict[str, Any]) -> None:
        result = estimate_token_savings(sample_snapshot, use_toon=True, use_compact_keys=True)
        assert result["savings_frac"] > 0.0
        assert result["encoding"] == "toon"

    def test_savings_keys_present(self, sample_snapshot: dict[str, Any]) -> None:
        result = estimate_token_savings(sample_snapshot)
        assert "json_chars" in result
        assert "encoded_chars" in result
        assert "savings_chars" in result
        assert "savings_frac" in result
        assert "encoding" in result

    def test_empty_snapshot_savings(self, empty_snapshot: dict[str, Any]) -> None:
        result = estimate_token_savings(empty_snapshot, use_toon=False, use_compact_keys=True)
        assert result["savings_frac"] >= 0.0  # May be zero for tiny payloads

    def test_savings_frac_bounded(self, sample_snapshot: dict[str, Any]) -> None:
        result = estimate_token_savings(sample_snapshot, use_toon=False, use_compact_keys=True)
        assert 0.0 <= result["savings_frac"] <= 1.0

    def test_token_estimates_available_with_tiktoken(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_snapshot: dict[str, Any],
    ) -> None:
        class _FakeEncoding:
            name = "cl100k_base"

            def encode(self, text: str) -> list[str]:
                return list(text)

        class _FakeTiktoken:
            @staticmethod
            def encoding_for_model(_model: str) -> _FakeEncoding:
                return _FakeEncoding()

            @staticmethod
            def get_encoding(_name: str) -> _FakeEncoding:
                return _FakeEncoding()

        monkeypatch.setitem(sys.modules, "tiktoken", _FakeTiktoken())
        result = estimate_token_savings(
            sample_snapshot,
            use_toon=False,
            use_compact_keys=True,
            model="gpt-4o-mini",
        )
        assert result["token_estimate_available"] is True
        assert result["token_estimate_model"] == "gpt-4o-mini"
        assert result["token_estimate_tokenizer"] == "cl100k_base"
        assert isinstance(result["json_tokens_estimate"], int)
        assert isinstance(result["encoded_tokens_estimate"], int)

    def test_token_estimates_degrade_when_tiktoken_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_snapshot: dict[str, Any],
    ) -> None:
        monkeypatch.setitem(sys.modules, "tiktoken", None)  # type: ignore[arg-type]
        result = estimate_token_savings(sample_snapshot, use_toon=False, use_compact_keys=True)
        assert result["token_estimate_available"] is False
        assert result["token_estimate_error"] == "tiktoken_unavailable"
        assert result["json_tokens_estimate"] is None


# ---------------------------------------------------------------------------
# _toons_available helper
# ---------------------------------------------------------------------------


class TestToonsAvailable:
    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_returns_true_when_installed(self) -> None:
        assert _toons_available() is True

    def test_returns_false_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        assert _toons_available() is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_encode_empty_dict(self) -> None:
        result = encode_for_llm({}, use_toon=False, use_compact_keys=True)
        assert json.loads(result) == {}

    def test_encode_non_serializable_uses_default_str(self) -> None:
        """The default=str fallback handles non-JSON-serializable values."""
        payload: dict[str, Any] = {"key": object()}
        result = encode_for_llm(payload, use_toon=False, use_compact_keys=False)
        parsed = json.loads(result)
        assert isinstance(parsed["key"], str)

    @pytest.mark.skipif(not _toons_installed(), reason="toons not installed")
    def test_toon_encoding_error_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, sample_snapshot: dict[str, Any]
    ) -> None:
        """If toons.dumps raises, we fall back to compact JSON."""
        mock_toons = MagicMock()
        mock_toons.dumps.side_effect = RuntimeError("toon encoding error")
        monkeypatch.setitem(sys.modules, "toons", mock_toons)

        result = encode_for_llm(sample_snapshot, use_toon=True, use_compact_keys=True)
        # Should still get valid JSON fallback
        parsed = json.loads(result)
        assert "_v" in parsed
