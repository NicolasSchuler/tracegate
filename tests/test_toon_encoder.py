"""Tests for llmdebug.toon_encoder module."""

from __future__ import annotations

import sys

import pytest

from llmdebug.toon_encoder import decode_snapshot_toon, encode_snapshot_toon


def _toons_available() -> bool:
    try:
        import toons

        return True
    except ImportError:
        return False


class TestEncodeSnapshotToon:
    def test_raises_import_error_when_toons_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="toons"):
            encode_snapshot_toon({"key": "value"})

    def test_import_error_message_mentions_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="pip install"):
            encode_snapshot_toon({})

    def test_import_error_message_mentions_llmdebug_toon_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match=r"llmdebug\[toon\]"):
            encode_snapshot_toon({"frames": []})

    @pytest.mark.skipif(not _toons_available(), reason="toons package not installed")
    def test_encode_returns_string(self) -> None:
        payload = {"name": "test", "frames": [{"file": "f.py", "line": 1}]}
        result = encode_snapshot_toon(payload)
        assert isinstance(result, str)
        assert len(result) > 0


class TestDecodeSnapshotToon:
    def test_raises_import_error_when_toons_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="toons"):
            decode_snapshot_toon('{"key": "value"}')

    def test_import_error_message_mentions_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="pip install"):
            decode_snapshot_toon("")

    def test_import_error_no_chained_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "toons", None)  # type: ignore[arg-type]
        try:
            decode_snapshot_toon("something")
        except ImportError as exc:
            assert exc.__cause__ is None

    @pytest.mark.skipif(not _toons_available(), reason="toons package not installed")
    def test_roundtrip(self) -> None:
        payload = {"name": "roundtrip", "exception": {"type": "ValueError"}}
        encoded = encode_snapshot_toon(payload)
        decoded = decode_snapshot_toon(encoded)
        assert decoded["name"] == "roundtrip"
        assert decoded["exception"]["type"] == "ValueError"
