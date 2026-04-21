"""Tests for log capture."""

import logging

import pytest

from llmdebug.log_capture import (
    LLMDebugLogHandler,
    get_recent_logs,
    install_log_handler,
    uninstall_log_handler,
)


@pytest.fixture(autouse=True)
def cleanup_handler():
    """Ensure handler is uninstalled after each test."""
    yield
    uninstall_log_handler()


def test_install_log_handler():
    """Test that log handler can be installed."""
    install_log_handler(max_records=10)

    # Log something
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.info("Test message")

    logs = get_recent_logs()
    assert len(logs) == 1
    assert logs[0]["message"] == "Test message"
    assert logs[0]["level"] == "INFO"


def test_get_recent_logs_returns_empty_without_handler():
    """Test that get_recent_logs returns empty list when no handler installed."""
    logs = get_recent_logs()
    assert logs == []


def test_log_capture_max_records():
    """Test that log records are limited to max_records."""
    install_log_handler(max_records=5)

    logger = logging.getLogger("test_max")
    logger.setLevel(logging.DEBUG)

    # Log more than max_records
    for i in range(10):
        logger.info(f"Message {i}")

    logs = get_recent_logs()
    assert len(logs) == 5

    # Should have the last 5 messages
    messages = [log["message"] for log in logs]
    assert messages == ["Message 5", "Message 6", "Message 7", "Message 8", "Message 9"]


def test_log_capture_with_limit_parameter():
    """Test that get_recent_logs respects the max_records parameter."""
    install_log_handler(max_records=10)

    logger = logging.getLogger("test_limit")
    logger.setLevel(logging.DEBUG)

    for i in range(10):
        logger.info(f"Message {i}")

    # Get only last 3
    logs = get_recent_logs(max_records=3)
    assert len(logs) == 3

    messages = [log["message"] for log in logs]
    assert messages == ["Message 7", "Message 8", "Message 9"]


def test_log_levels_captured():
    """Test that all log levels are captured."""
    install_log_handler()

    logger = logging.getLogger("test_levels")
    logger.setLevel(logging.DEBUG)

    logger.debug("Debug msg")
    logger.info("Info msg")
    logger.warning("Warning msg")
    logger.error("Error msg")

    logs = get_recent_logs()
    assert len(logs) == 4

    levels = [log["level"] for log in logs]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR"]


def test_log_metadata_captured():
    """Test that log metadata is captured correctly."""
    install_log_handler()

    logger = logging.getLogger("test_meta")
    logger.setLevel(logging.DEBUG)
    logger.info("Test metadata")

    logs = get_recent_logs()
    assert len(logs) == 1

    log = logs[0]
    assert "timestamp" in log
    assert log["level"] == "INFO"
    assert log["logger"] == "test_meta"
    assert log["message"] == "Test metadata"
    assert "filename" in log
    assert "lineno" in log


def test_log_with_exception():
    """Test that exception info is captured in logs."""
    install_log_handler()

    logger = logging.getLogger("test_exc")
    logger.setLevel(logging.DEBUG)

    try:
        raise ValueError("test error")
    except ValueError:
        logger.exception("An error occurred")

    logs = get_recent_logs()
    assert len(logs) == 1
    assert logs[0]["exc_type"] == "ValueError"


def test_install_idempotent():
    """Test that installing multiple times is safe."""
    install_log_handler(max_records=5)
    install_log_handler(max_records=5)  # Should not create duplicate handlers

    logger = logging.getLogger("test_idem")
    logger.setLevel(logging.DEBUG)
    logger.info("Test")

    logs = get_recent_logs()
    # Should only have one message, not duplicated
    assert len(logs) == 1


def test_install_updates_max_records():
    """Test that reinstalling updates max_records."""
    install_log_handler(max_records=10)

    logger = logging.getLogger("test_update")
    logger.setLevel(logging.DEBUG)

    for i in range(10):
        logger.info(f"Message {i}")

    # Update to smaller max_records
    install_log_handler(max_records=5)

    # Should still have 5 messages (truncated to new max)
    logs = get_recent_logs()
    assert len(logs) == 5


def test_uninstall_clears_logs():
    """Test that uninstalling removes the handler."""
    install_log_handler()

    logger = logging.getLogger("test_uninstall")
    logger.setLevel(logging.DEBUG)
    logger.info("Before uninstall")

    uninstall_log_handler()

    # After uninstall, no logs should be returned
    logs = get_recent_logs()
    assert logs == []


def test_logs_oldest_first():
    """Test that logs are returned oldest first."""
    install_log_handler()

    logger = logging.getLogger("test_order")
    logger.setLevel(logging.DEBUG)

    logger.info("First")
    logger.info("Second")
    logger.info("Third")

    logs = get_recent_logs()
    messages = [log["message"] for log in logs]
    assert messages == ["First", "Second", "Third"]


def test_multiple_loggers():
    """Test that logs from multiple loggers are captured."""
    install_log_handler()

    logger1 = logging.getLogger("logger1")
    logger1.setLevel(logging.DEBUG)
    logger2 = logging.getLogger("logger2")
    logger2.setLevel(logging.DEBUG)

    logger1.info("From logger1")
    logger2.info("From logger2")

    logs = get_recent_logs()
    assert len(logs) == 2

    loggers = [log["logger"] for log in logs]
    assert "logger1" in loggers
    assert "logger2" in loggers


def test_handler_clear_removes_records():
    handler = LLMDebugLogHandler(max_records=5)
    record = logging.LogRecord(
        name="clear_test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    handler.emit(record)
    assert len(handler.get_records()) == 1

    handler.clear()
    assert handler.get_records() == []


def test_emit_calls_handle_error_when_formatting_fails(monkeypatch):
    handler = LLMDebugLogHandler(max_records=5)
    handler.setFormatter(logging.Formatter("%(missing_field)s"))

    called = {"value": False}
    monkeypatch.setattr(handler, "handleError", lambda _record: called.__setitem__("value", True))

    record = logging.LogRecord(
        name="broken_format",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    handler.emit(record)

    assert called["value"] is True


def test_uninstall_is_noop_when_not_installed():
    uninstall_log_handler()
    assert get_recent_logs() == []
