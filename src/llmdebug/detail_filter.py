"""Layered snapshot disclosure — read-time filter for detail levels.

Pure filtering module with no I/O. Operates on expanded-key dicts only.
Three levels:
  - crash:   Exception + crash frame (~2K tokens)
  - full:    All frames + traceback (~5K tokens)
  - context: Everything (~10K tokens)
"""

from __future__ import annotations

import copy
from typing import Any, Literal, cast

from ._debug_session import coerce_crash_frame_index

DetailLevel = Literal["crash", "full", "context"]

# Keys included at the "crash" layer (minimal)
_LAYER_CRASH_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "name",
        "timestamp_utc",
        "exception",
        "crash_frame_index",
        "llmdebug_version",
        "frames",
    }
)

# Keys added at the "full" layer
_LAYER_FULL_KEYS: frozenset[str] = _LAYER_CRASH_KEYS | frozenset(
    {
        "traceback",
        "capture_config",
        "_capture_warnings",
    }
)


def filter_snapshot(snapshot: dict[str, Any], detail: DetailLevel = "crash") -> dict[str, Any]:
    """Filter a snapshot to the requested detail level.

    Args:
        snapshot: Expanded-key snapshot dict (not compact).
        detail: One of "crash", "full", "context".

    Returns:
        Filtered copy of the snapshot. Does not mutate the original.

    Raises:
        ValueError: If snapshot appears to use compact keys.
    """
    # Guard against compact snapshots
    if "__llmdebug_compact__" in snapshot or "_exc" in snapshot or "_fr" in snapshot:
        raise ValueError(
            "filter_snapshot requires expanded-key snapshots. "
            "Use get_latest_snapshot() or expand_keys() first."
        )

    if detail == "context":
        # Everything — shallow copy
        return dict(snapshot)

    if detail == "full":
        result: dict[str, Any] = {}
        for key in snapshot:
            if key in _LAYER_FULL_KEYS:
                result[key] = snapshot[key]
        return result

    # detail == "crash"
    result = {}
    for key in _LAYER_CRASH_KEYS:
        if key in snapshot:
            result[key] = snapshot[key]

    # Trim frames to only the crash frame
    frames = snapshot.get("frames", [])
    if isinstance(frames, list) and frames:
        frames_list = cast(list[Any], frames)
        crash_idx = coerce_crash_frame_index(snapshot)
        if 0 <= crash_idx < len(frames_list):
            result["frames"] = [copy.deepcopy(frames_list[crash_idx])]
        else:  # pragma: no cover
            result["frames"] = [copy.deepcopy(frames_list[0])]
        result["crash_frame_index"] = 0

    return result
