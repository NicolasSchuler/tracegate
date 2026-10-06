"""Shared warning emitter helpers."""

from __future__ import annotations

import sys


def emit_warning(message: str) -> None:
    """Emit a standardized llmdebug warning message to stderr."""
    sys.stderr.write(f"llmdebug: warning: {message}\n")
