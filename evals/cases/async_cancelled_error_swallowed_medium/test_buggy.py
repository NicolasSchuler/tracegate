from buggy import run


def test_cancellation_propagates() -> None:
    status, log = run()
    assert status == "cancelled", f"expected cancellation, got {status!r}"
    assert "suppressed" not in log, "CancelledError was suppressed instead of propagating"
