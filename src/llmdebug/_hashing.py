"""Stable non-cryptographic hashing helpers."""

from __future__ import annotations

from typing import Any

import xxhash

CONTENT_HASH_ALGORITHM = "xxh3_128"
SHORT_HASH_ALGORITHM = "xxh64"


def xxh3_128_hexdigest_bytes(data: bytes) -> str:
    """Return a stable 128-bit xxhash digest for raw bytes."""
    return xxhash.xxh3_128_hexdigest(data)


def xxh3_128_hexdigest_text(text: str) -> str:
    """Return a stable 128-bit xxhash digest for UTF-8 text."""
    return xxh3_128_hexdigest_bytes(text.encode("utf-8"))


def xxh64_hexdigest_text(text: str) -> str:
    """Return a stable 64-bit xxhash digest for UTF-8 text."""
    return xxhash.xxh64_hexdigest(text.encode("utf-8"))


def new_xxh3_128() -> Any:
    """Create a streaming 128-bit xxhash instance."""
    return xxhash.xxh3_128()
