"""Derive independent, reproducible seeds from a run seed (review findings POL-05, POL-23)."""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *parts: object) -> int:
    """Return a 63-bit seed for the stream named by *parts* under *base_seed*.

    Distinct names give unrelated streams, so policy controllers no longer share one
    exploration stream, and a per-decision key (case, condition, attempt) makes draws
    independent of how worker threads interleave.
    """
    material = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
