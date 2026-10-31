"""Stable non-cryptographic hashing helpers."""

from __future__ import annotations

from typing import Any

import xxhash

CONTENT_HASH_ALGORITHM = "xxh3_128"
SHORT_HASH_ALGORITHM = "xxh64"


def xxh3_128_hexdigest_bytes(data: bytes) -> str:
    """Return a stable 128-bit xxhash digest for raw bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Lowercase hexadecimal XXH3-128 digest.
    """
    return xxhash.xxh3_128_hexdigest(data)


def xxh3_128_hexdigest_text(text: str) -> str:
    """Return a stable 128-bit xxhash digest for UTF-8 text.

    Args:
        text: Text to hash after UTF-8 encoding.

    Returns:
        Lowercase hexadecimal XXH3-128 digest.
    """
    return xxh3_128_hexdigest_bytes(text.encode("utf-8"))


def xxh64_hexdigest_text(text: str) -> str:
    """Return a stable 64-bit xxhash digest for UTF-8 text.

    Args:
        text: Text to hash after UTF-8 encoding.

    Returns:
        Lowercase hexadecimal XXH64 digest.
    """
    return xxhash.xxh64_hexdigest(text.encode("utf-8"))


def new_xxh3_128() -> Any:
    """Create a streaming 128-bit xxhash instance.

    Returns:
        Fresh XXH3-128 hasher that accepts incremental ``update()`` calls.
    """
    return xxhash.xxh3_128()
