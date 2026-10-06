"""Dedicated tests for llmdebug.compact_keys module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

import llmdebug.compact_keys as compact_keys_module
from llmdebug.compact_keys import (
    KEY_MAP,
    LEGACY_REVERSE_KEY_MAP,
    REVERSE_KEY_MAP,
    USER_DATA_KEYS,
    compact_keys,
    expand_keys,
    is_compact_snapshot,
)

# ---------------------------------------------------------------------------
# KEY_MAP / REVERSE_KEY_MAP integrity
# ---------------------------------------------------------------------------


class TestKeyMapIntegrity:
    """Structural guarantees on the abbreviation tables."""

    def test_key_map_is_bijective(self) -> None:
        assert len(KEY_MAP) == len(set(KEY_MAP.values()))
        assert len(REVERSE_KEY_MAP) == len(KEY_MAP) + len(LEGACY_REVERSE_KEY_MAP)

    def test_retired_short_keys_are_not_reused_by_current_entries(self) -> None:
        """CAP-17: reusing one would silently decode old snapshots to the wrong key."""
        assert not set(LEGACY_REVERSE_KEY_MAP) & set(KEY_MAP.values())

    def test_reverse_map_inverts_key_map(self) -> None:
        for full_key, short_key in KEY_MAP.items():
            assert REVERSE_KEY_MAP[short_key] == full_key

    def test_all_short_keys_start_with_underscore(self) -> None:
        for short in KEY_MAP.values():
            assert short.startswith("_"), f"Short key {short!r} should start with '_'"

    def test_user_data_keys_cover_both_forms(self) -> None:
        """Every full user-data key has its compact twin in USER_DATA_KEYS."""
        full_keys_in_set = {k for k in USER_DATA_KEYS if not k.startswith("_")}
        for fk in full_keys_in_set:
            if fk in KEY_MAP:
                assert KEY_MAP[fk] in USER_DATA_KEYS, (
                    f"Full key {fk!r} is in USER_DATA_KEYS but its compact form "
                    f"{KEY_MAP[fk]!r} is not"
                )


# ---------------------------------------------------------------------------
# Module-level duplicate-value guard (line 130-134)
# ---------------------------------------------------------------------------


class TestDuplicateShortKeyGuard:
    """Cover the ValueError raised when KEY_MAP has duplicate short-key values."""

    def test_duplicate_short_key_raises_on_import(self, tmp_path: Path) -> None:
        """Load a temp module copy with a duplicate short key and execute it."""
        assert compact_keys_module.__file__ is not None
        source = Path(compact_keys_module.__file__).read_text(encoding="utf-8")
        bad_source = source.replace('"timestamp_utc": "_ts",', '"timestamp_utc": "_v",', 1)
        assert bad_source != source

        temp_module = tmp_path / "bad_compact_keys.py"
        temp_module.write_text(bad_source, encoding="utf-8")

        spec = importlib.util.spec_from_file_location("bad_compact_keys", temp_module)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        with pytest.raises(ValueError, match="duplicate short-key values"):
            spec.loader.exec_module(module)

    def test_actual_key_map_passes_guard(self) -> None:
        """The shipped KEY_MAP must not trigger the guard."""
        reverse = {v: k for k, v in KEY_MAP.items()}
        assert len(reverse) == len(KEY_MAP)


# ---------------------------------------------------------------------------
# compact_keys / expand_keys — public API
# ---------------------------------------------------------------------------


class TestCompactKeys:
    """Tests for compact_keys()."""

    def test_empty_dict(self) -> None:
        assert compact_keys({}) == {}

    def test_empty_list(self) -> None:
        assert compact_keys([]) == []

    def test_primitive_passthrough(self) -> None:
        assert compact_keys(42) == 42
        assert compact_keys("hello") == "hello"
        assert compact_keys(None) is None
        assert compact_keys(True) is True

    def test_simple_dict(self) -> None:
        result = compact_keys({"schema_version": 1, "timestamp_utc": "t"})
        assert result == {"_v": 1, "_ts": "t"}

    def test_nested_dict(self) -> None:
        data: dict[str, Any] = {"exception": {"type": "ValueError", "message": "boom"}}
        result = compact_keys(data)
        assert result == {"_exc": {"_t": "ValueError", "_msg": "boom"}}

    def test_list_of_dicts(self) -> None:
        data: dict[str, list[dict[str, Any]]] = {
            "frames": [{"file": "a.py", "line": 1}],
        }
        result = compact_keys(data)
        assert result == {"_fr": [{"_f": "a.py", "_l": 1}]}

    def test_unknown_keys_preserved(self) -> None:
        result = compact_keys({"custom_field": 99})
        assert result == {"custom_field": 99}

    def test_mixed_known_unknown_keys(self) -> None:
        data = {"schema_version": 1, "my_field": "hi"}
        result = compact_keys(data)
        assert result == {"_v": 1, "my_field": "hi"}


class TestExpandKeys:
    """Tests for expand_keys()."""

    def test_empty_dict(self) -> None:
        assert expand_keys({}) == {}

    def test_empty_list(self) -> None:
        assert expand_keys([]) == []

    def test_primitive_passthrough(self) -> None:
        assert expand_keys(42) == 42
        assert expand_keys("hello") == "hello"
        assert expand_keys(None) is None

    def test_simple_dict(self) -> None:
        result = expand_keys({"_v": 1, "_ts": "t"})
        assert result == {"schema_version": 1, "timestamp_utc": "t"}

    def test_nested_expansion(self) -> None:
        data: dict[str, dict[str, str]] = {"_exc": {"_t": "ValueError", "_msg": "boom"}}
        result = expand_keys(data)
        assert result == {"exception": {"type": "ValueError", "message": "boom"}}

    def test_unknown_short_keys_preserved(self) -> None:
        result = expand_keys({"_unknown_key": "val"})
        assert result == {"_unknown_key": "val"}


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify compact → expand and expand → compact are inverses."""

    @pytest.mark.parametrize(
        "original",
        [
            {"schema_version": 1},
            {"frames": [{"file": "a.py", "line": 1, "function": "f"}]},
            {
                "exception": {"type": "TypeError", "message": "x"},
                "traceback": "Traceback ...",
            },
            {"git": {"commit": "abc", "branch": "main", "dirty": False}},
            {},
        ],
        ids=["single-key", "frames-list", "exception-traceback", "git", "empty"],
    )
    def test_compact_expand_roundtrip(self, original: dict[str, Any]) -> None:
        assert expand_keys(compact_keys(original)) == original

    def test_roundtrip_with_deeply_nested_data(self) -> None:
        original: dict[str, Any] = {
            "frames": [
                {
                    "file": "/a.py",
                    "line": 10,
                    "locals": {"my_var": {"shape": [3, 4]}},
                    "source": {"start": 8, "end": 12, "snippet": "x = 1"},
                }
            ],
            "env": {"python": "3.12", "platform": "linux"},
        }
        assert expand_keys(compact_keys(original)) == original


# ---------------------------------------------------------------------------
# User-data protection
# ---------------------------------------------------------------------------


class TestUserDataProtection:
    """Keys inside user-data subtrees must never be transformed."""

    def test_locals_keys_not_compacted(self) -> None:
        data: dict[str, list[dict[str, Any]]] = {
            "frames": [
                {
                    "file": "t.py",
                    "locals": {"type": "my_type", "message": "my_msg"},
                }
            ]
        }
        result = compact_keys(data)
        frame = result["_fr"][0]
        # "type" inside locals must stay as "type", not become "_t"
        assert frame["_loc"]["type"] == "my_type"
        assert frame["_loc"]["message"] == "my_msg"

    def test_args_keys_not_compacted(self) -> None:
        data: dict[str, list[dict[str, Any]]] = {
            "frames": [{"file": "t.py", "args": {"line": 99}}],
        }
        result = compact_keys(data)
        assert result["_fr"][0]["_args"]["line"] == 99

    def test_params_keys_not_compacted(self) -> None:
        data: dict[str, dict[str, str]] = {"pytest": {"params": {"type": "int"}}}
        result = compact_keys(data)
        assert result["_pt"]["_p"]["type"] == "int"

    def test_locals_meta_roundtrip(self) -> None:
        data = {
            "frames": [
                {
                    "file": "t.py",
                    "locals_meta": {"type": {"type": "builtins.str"}},
                }
            ]
        }
        assert expand_keys(compact_keys(data)) == data

    def test_fixtures_keys_not_compacted(self) -> None:
        data: dict[str, dict[str, dict[str, str]]] = {
            "pytest": {"fixtures": {"schema_version": "repr_str"}},
        }
        result = compact_keys(data)
        # "schema_version" inside fixtures must stay unchanged
        assert result["_pt"]["_fix"]["schema_version"] == "repr_str"

    def test_in_user_data_flag_skips_all_transforms(self) -> None:
        data = {"schema_version": 1, "type": "X"}
        result = compact_keys(data, _in_user_data=True)
        assert result == data  # No key renaming at all

    def test_expand_user_data_flag(self) -> None:
        data = {"_v": 1, "_t": "X"}
        result = expand_keys(data, _in_user_data=True)
        assert result == data  # No key renaming


# ---------------------------------------------------------------------------
# is_compact_snapshot
# ---------------------------------------------------------------------------


class TestIsCompactSnapshot:
    """Detection heuristics for compact vs full key format."""

    def test_explicit_marker(self) -> None:
        assert is_compact_snapshot({"__llmdebug_compact__": 1}) is True

    def test_marker_wrong_value_still_false(self) -> None:
        assert is_compact_snapshot({"__llmdebug_compact__": 0}) is False

    def test_full_keys_detected(self) -> None:
        assert is_compact_snapshot({"schema_version": 1, "frames": []}) is False

    def test_legacy_compact_structural_keys(self) -> None:
        assert is_compact_snapshot({"_fr": [], "_exc": {}}) is True

    def test_debug_session_structural_keys(self) -> None:
        assert is_compact_snapshot({"_sess": {}, "_snap": {}}) is True

    def test_empty_dict_is_not_compact(self) -> None:
        assert is_compact_snapshot({}) is False

    def test_user_payload_not_false_positive(self) -> None:
        """A dict with only _v (no structural keys) should not look compact."""
        assert is_compact_snapshot({"_v": "some_user_value"}) is False

    def test_single_full_key_is_not_compact(self) -> None:
        assert is_compact_snapshot({"name": "test"}) is False

    def test_mixed_full_and_compact_prefers_full(self) -> None:
        """If full-key structural keys are present alongside short ones, treat as full."""
        data = {"schema_version": 1, "frames": [], "_exc": {}}
        assert is_compact_snapshot(data) is False


def test_every_key_map_entry_can_apply() -> None:
    """CAP-17: entries for keys that only occur in user-data subtrees were dead."""
    summary_only = {"__array__", "__dataframe__", "__series__", "shape", "dtype", "stats", "head"}

    assert not summary_only & set(KEY_MAP)


def test_legacy_compact_extra_context_still_expands(tmp_path: Path) -> None:
    """CAP-17: compaction skipped locals and args, not user-supplied extra context."""
    from llmdebug import _json as json
    from llmdebug._snapshot_loader import load_snapshot_from_file
    from llmdebug.output import get_latest_snapshot

    # As an earlier release wrote it for extra={"stats": {"min": 1}, "device": "cuda"}.
    legacy = {
        "__llmdebug_compact__": True,
        "_v": "2.1",
        "_n": "legacy",
        "_exc": {"_t": "ValueError", "_msg": "boom"},
        "_fr": [],
        "_ctx": {"_st": {"_mn": 1, "_mx": 9}, "_dev": "cuda"},
    }
    expected_context = {"stats": {"min": 1, "max": 9}, "device": "cuda"}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    (tmp_path / "latest.json").write_text(json.dumps(legacy), encoding="utf-8")

    assert load_snapshot_from_file(path)["context"] == expected_context
    latest = get_latest_snapshot(str(tmp_path))
    assert latest is not None
    assert latest["context"] == expected_context
