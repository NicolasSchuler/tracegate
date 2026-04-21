"""TOON format encoder for llmdebug snapshots.

TOON (Token-Oriented Object Notation) provides better compression for
uniform arrays, making it ideal for debug snapshots with repeated frame
structures.

This module requires the optional 'toons' package:
    pip install 'llmdebug[toon]'
"""

from __future__ import annotations

from typing import Any


def encode_snapshot_toon(payload: dict[str, Any]) -> str:
    """Encode snapshot to TOON format.

    Uses TOON's tabular format for uniform arrays (frames, source snippets)
    and standard encoding for nested/irregular data (locals, exception).

    Args:
        payload: Snapshot data dictionary

    Returns:
        TOON-encoded string

    Raises:
        ImportError: If toons package is not installed
    """
    try:
        import toons  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError(
            "TOON format requires the 'toons' package. Install with: pip install 'llmdebug[toon]'"
        ) from None

    # TOON handles uniform arrays (like frames) efficiently
    # No preprocessing needed - the library auto-detects tabular structures
    return toons.dumps(payload)  # type: ignore[no-any-return]


def decode_snapshot_toon(content: str) -> dict[str, Any]:
    """Decode TOON-formatted snapshot.

    Args:
        content: TOON-encoded string

    Returns:
        Snapshot data dictionary

    Raises:
        ImportError: If toons package is not installed
    """
    try:
        import toons  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError(
            "TOON format requires the 'toons' package. Install with: pip install 'llmdebug[toon]'"
        ) from None

    return toons.loads(content)  # type: ignore[no-any-return]
