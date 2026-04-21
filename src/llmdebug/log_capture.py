"""Log capture for debug snapshots."""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


class LLMDebugLogHandler(logging.Handler):
    """Memory handler that stores recent log records in a thread-safe deque."""

    def __init__(self, max_records: int = 100):
        super().__init__()
        self.max_records = max_records
        self.records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Store log record as a dict."""
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "filename": record.filename,
                "lineno": record.lineno,
            }
            # Include exception info if present
            if record.exc_info and record.exc_info[0] is not None:
                entry["exc_type"] = record.exc_info[0].__name__
            with self._lock:
                self.records.append(entry)
        except Exception:
            # Use standard logging error reporting rather than silently dropping
            self.handleError(record)

    def get_records(self, max_records: int | None = None) -> list[dict[str, Any]]:
        """Get recent log records, oldest first."""
        with self._lock:
            records = list(self.records)
        if max_records is not None and len(records) > max_records:
            return records[-max_records:]
        return records

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self.records.clear()


# Global handler instance
_handler: LLMDebugLogHandler | None = None
_handler_lock = threading.Lock()


def install_log_handler(max_records: int = 100) -> None:
    """Install the log handler on the root logger.

    This function is idempotent - calling it multiple times is safe.
    Call this early in your application to capture logs before crashes.

    Args:
        max_records: Maximum number of log records to keep in memory.
    """
    global _handler
    with _handler_lock:
        if _handler is not None:
            # Already installed, just update max_records if different
            if _handler.max_records != max_records:
                # Use handler's internal lock to avoid TOCTOU race
                with _handler._lock:  # pyright: ignore[reportPrivateUsage]
                    _handler.records = deque(_handler.records, maxlen=max_records)
                    _handler.max_records = max_records
            return

        _handler = LLMDebugLogHandler(max_records=max_records)
        _handler.setLevel(logging.DEBUG)  # Capture all levels
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(_handler)


def get_recent_logs(max_records: int | None = None) -> list[dict[str, Any]]:
    """Get recent log records for inclusion in snapshots.

    Args:
        max_records: Maximum number of records to return (oldest first).
                    None returns all stored records.

    Returns:
        List of log record dicts, oldest first.
    """
    with _handler_lock:
        if _handler is None:
            return []
        return _handler.get_records(max_records)


def uninstall_log_handler() -> None:
    """Remove the log handler from the root logger.

    Primarily useful for testing cleanup.
    """
    global _handler
    with _handler_lock:
        if _handler is not None:
            logging.getLogger().removeHandler(_handler)
            _handler = None
