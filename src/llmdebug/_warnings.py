"""The one channel for llmdebug's own diagnostics: the ``llmdebug.diagnostics`` logger (X-07).

Warnings go to stderr as ``llmdebug: warning: <message>`` by default, as before.
Applications can retarget or silence them by configuring that logger. It doesn't
propagate to the root logger, so ``log_capture`` doesn't record llmdebug's own
diagnostics into snapshots; it is a sibling of ``llmdebug.mcp``, which is unaffected.
Capture keeps its per-snapshot ``_capture_warnings`` field and also logs each entry
here at DEBUG level.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "llmdebug.diagnostics"


class _StderrHandler(logging.Handler):
    """Logging handler that writes ``llmdebug: <level>: <message>`` to stderr.

    ``sys.stderr`` is looked up at emit time, so later redirections are honoured.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write *record* to the current stderr, routing failures to ``handleError``."""
        try:
            sys.stderr.write(f"llmdebug: {record.levelname.lower()}: {record.getMessage()}\n")
        except Exception:
            self.handleError(record)


_HANDLER_NAME = "llmdebug.diagnostics.stderr"


def _configure_logger() -> logging.Logger:
    """Return the diagnostics logger, attaching the stderr handler at most once.

    Sets the level to WARNING when it is unset and stops propagation to the root logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    # Found by name, not class: a re-imported module defines a new _StderrHandler class,
    # and an isinstance check would add a second handler and print every warning twice.
    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = _StderrHandler()
        handler.set_name(_HANDLER_NAME)
        logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.WARNING)
    logger.propagate = False
    return logger


logger = _configure_logger()


def emit_warning(message: str) -> None:
    """Log *message* as an llmdebug warning on the diagnostics logger.

    By default it is written to stderr as ``llmdebug: warning: <message>``; see the
    module docstring for how to retarget or silence it.

    Args:
        message: The warning text, logged verbatim.
    """
    logger.warning("%s", message)


def emit_debug(message: str) -> None:
    """Log *message* at DEBUG level on the diagnostics logger.

    Hidden unless the ``llmdebug.diagnostics`` logger's level is lowered to DEBUG.

    Args:
        message: The diagnostic text, logged verbatim.
    """
    logger.debug("%s", message)


def emit_capture_diagnostics(messages: list[str]) -> None:
    """Surface buffered capture diagnostics for ``SnapshotConfig(debug=True)`` (X-07).

    Capture used to write these straight to stderr, so configuring the diagnostics
    logger neither silenced nor retargeted them. They go through the logger now, at
    WARNING so the debug flag still shows them with the default level -- and only when
    DEBUG is off, because each one was already logged there as it was recorded.

    Args:
        messages: Capture warnings collected for one snapshot; each is logged with a
            ``capture:`` prefix.
    """
    if logger.isEnabledFor(logging.DEBUG):
        return
    for message in messages:
        logger.warning("capture: %s", message)
