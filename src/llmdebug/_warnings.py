"""Shared warning emitter helpers."""

from __future__ import annotations  # pragma: no cover

import sys  # pragma: no cover


def emit_warning(message: str) -> None:  # pragma: no cover - def line executes at import time
    """Emit a standardized llmdebug warning message to stderr."""
    sys.stderr.write(f"llmdebug: warning: {message}\n")
