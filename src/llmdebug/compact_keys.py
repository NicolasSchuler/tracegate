"""Key abbreviation mappings for compact JSON output format."""

from __future__ import annotations

from typing import Any, cast

# Full key -> Short key mapping
# Organized by category for maintainability
# IMPORTANT: Short keys must be unique and unlikely to appear as user variable names
# We use prefixed keys (_X format) to avoid collisions with user data
KEY_MAP: dict[str, str] = {
    # Top-level keys
    "schema_version": "_v",
    "timestamp_utc": "_ts",
    "exception": "_exc",
    "traceback": "_tb",
    "frames": "_fr",
    "crash_frame_index": "_cfi",
    "capture_config": "_cfg",
    "llmdebug_version": "_ldv",
    "env": "_env",
    "name": "_n",
    "kind": "_kind",
    "session": "_sess",
    "snapshot": "_snap",
    "trace": "_trc",
    "eval": "_eval",
    "pytest": "_pt",
    "git": "_git",
    "async": "_async",
    "recent_logs": "_logs",
    # Frame keys
    "file": "_f",
    "file_rel": "_frel",
    "line": "_l",
    "function": "_fn",
    "module": "_mod",
    "code": "_c",
    "source": "_src",
    "locals": "_loc",
    "locals_meta": "_lm",
    "locals_truncated": "_lt",
    "locals_truncated_keys": "_ltk",
    "args": "_args",
    "args_meta": "_am",
    "arg_names": "_an",
    "is_coroutine": "_coro",
    # Exception keys
    "type": "_t",
    "qualified_type": "_qt",
    "message": "_msg",
    "suppress_context": "_sc",
    "error_category": "_ecat",
    "category": "_cat",
    "suggestion": "_sug",
    "cause": "_cau",
    "context": "_ctx",
    # Source keys
    "start": "_s",
    "end": "_e",
    "snippet": "_snip",
    "lineno": "_ln",
    # No array/DataFrame summary keys: they only occur inside locals and args, which are
    # never compacted, so those 16 entries could never apply (CAP-17).
    # Env keys
    "python": "_py",
    "executable": "_exe",
    "platform": "_plat",
    "cwd": "_wd",
    "argv": "_av",
    # Git keys
    "commit": "_cm",
    "commit_full": "_cmf",
    "branch": "_br",
    "dirty": "_d",
    # Log keys
    "level": "_lv",
    "logger": "_lgr",
    "filename": "_fname",
    "created": "_cr",
    # Pytest keys
    "nodeid": "_nid",
    "outcome": "_out",
    "when": "_w",
    "longrepr": "_lr",
    "capstdout": "_cout",
    "capstderr": "_cerr",
    "params": "_p",
    "repro": "_rp",
    # Pytest keys (enhanced context)
    "fixtures": "_fix",
    "markers": "_mk",
    "param_indices": "_pi",
    "xdist_worker": "_xw",
    "timeout_config": "_toc",
    "rerun_count": "_rc",
    "flaky": "_fl",
    "seconds": "_sec",
    "kwargs": "_kw",
    # Coverage keys
    "coverage": "_cov",
    "executed_lines": "_el",
    "missing_lines": "_ml",
    "line_rate": "_lr2",
    "line_rate_unknown": "_lru",
    "partial_branches": "_pb",
    "executable_count": "_ec",
    "executed_count": "_xc",
    "missing_count": "_mc",
}

# Short keys earlier releases wrote that current captures never produce. Decoding has to
# keep understanding them: compaction skips locals and args, but not user-supplied extra
# context, so a snapshot captured with extra={"stats": {"min": 1}, "device": "cuda"} holds
# `_st`, `_mn` and `_dev` (CAP-17).
LEGACY_REVERSE_KEY_MAP: dict[str, str] = {
    "_arr": "__array__",
    "_df": "__dataframe__",
    "_ser": "__series__",
    "_typ": "__type__",
    "_sh": "shape",
    "_dt": "dtype",
    "_hd": "head",
    "_ht": "head_truncated",
    "_dev": "device",
    "_rg": "requires_grad",
    "_anom": "anomalies",
    "_st": "stats",
    "_mu": "mean",
    "_sd": "std",
    "_mn": "min",
    "_mx": "max",
}

# Reverse mapping for decoding: current keys plus the retired ones old files still carry.
REVERSE_KEY_MAP: dict[str, str] = {
    **LEGACY_REVERSE_KEY_MAP,
    **{v: k for k, v in KEY_MAP.items()},
}

# One guard for both failure modes: a short key used twice in KEY_MAP, and a KEY_MAP entry
# that reuses a retired one (which would decode old snapshots to the wrong key).
if len(REVERSE_KEY_MAP) != len(KEY_MAP) + len(LEGACY_REVERSE_KEY_MAP):
    raise ValueError(
        f"KEY_MAP has duplicate short-key values, or reuses a retired one: "
        f"{len(KEY_MAP)} current and {len(LEGACY_REVERSE_KEY_MAP)} retired keys map to "
        f"only {len(REVERSE_KEY_MAP)} unique values"
    )

# Keys that contain user data (locals, args) - don't transform their contents
USER_DATA_KEYS = {
    # Full keys
    "locals",
    "locals_meta",
    "args",
    "args_meta",
    "params",
    # Compact keys
    "_loc",
    "_lm",
    "_args",
    "_am",
    "_p",
    # Enhanced pytest context (fixture/marker names are user-chosen)
    "fixtures",
    "markers",
    "_fix",
    "_mk",
}


def _transform_keys(
    data: dict[str, Any] | list[Any] | Any,
    key_map: dict[str, str],
    _in_user_data: bool = False,
) -> dict[str, Any] | list[Any] | Any:
    """Recursively remap dict keys using the given mapping.

    Keys inside user-data subtrees (locals, args, params, etc.) are left
    untouched so that user variable names are never mangled.

    Args:
        data: Data structure to transform (dict, list, or primitive).
        key_map: Mapping from current key names to desired key names.
        _in_user_data: Internal flag -- True when inside a user-data subtree.

    Returns:
        Same structure with keys replaced according to *key_map*.
    """
    if isinstance(data, dict):
        data_dict = cast(dict[str, Any], data)
        result: dict[str, Any] = {}
        for k, v in data_dict.items():
            new_key = k if _in_user_data else key_map.get(k, k)
            is_user_data = _in_user_data or k in USER_DATA_KEYS or new_key in USER_DATA_KEYS
            result[new_key] = _transform_keys(v, key_map, _in_user_data=is_user_data)
        return result
    if isinstance(data, list):
        data_list = cast(list[Any], data)
        return [_transform_keys(item, key_map, _in_user_data=_in_user_data) for item in data_list]
    return data


def compact_keys(
    data: dict[str, Any] | list[Any] | Any,
    _in_user_data: bool = False,
) -> dict[str, Any] | list[Any] | Any:
    """Recursively replace keys with their short ``KEY_MAP`` versions.

    Keys inside user-data subtrees (locals, args, params, fixtures, markers) are kept.

    Args:
        data: Snapshot structure to compact (dict, list, or primitive).
        _in_user_data: Internal flag; True treats *data* as a user-data subtree.

    Returns:
        A new structure with framework keys abbreviated.
    """
    return _transform_keys(data, KEY_MAP, _in_user_data=_in_user_data)


def expand_keys(
    data: dict[str, Any] | list[Any] | Any,
    _in_user_data: bool = False,
) -> dict[str, Any] | list[Any] | Any:
    """Recursively restore original keys from short versions.

    Retired short keys from ``LEGACY_REVERSE_KEY_MAP`` are expanded as well.

    Args:
        data: Compact snapshot structure (dict, list, or primitive).
        _in_user_data: Internal flag; True treats *data* as a user-data subtree.

    Returns:
        A new structure with full key names.
    """
    return _transform_keys(data, REVERSE_KEY_MAP, _in_user_data=_in_user_data)


def is_compact_snapshot(data: dict[str, Any]) -> bool:
    """Detect if a snapshot uses compact key format.

    Prefers an explicit format marker written by recent versions.
    Falls back to a conservative heuristic for backward compatibility with
    compact snapshots produced before the marker existed.

    Args:
        data: Top-level snapshot dict as loaded from disk.

    Returns:
        True if the compact marker is set, or if no canonical full keys are present
        and at least one compact structural key is.
    """
    if data.get("__llmdebug_compact__") == 1:
        return True

    # If canonical full snapshot keys are present, treat as non-compact.
    if {"schema_version", "frames", "exception", "timestamp_utc", "name"} & set(data.keys()):
        return False

    # Legacy compact snapshots may not have a marker. Require structural keys,
    # not only "_v", to avoid false positives on user payloads.
    legacy_structural_keys = {"_fr", "_exc", "_tb", "_cfg"}
    debug_session_structural_keys = {"_sess", "_snap"}
    return bool((legacy_structural_keys | debug_session_structural_keys) & set(data.keys()))
