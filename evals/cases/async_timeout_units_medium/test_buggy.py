from buggy import run_fetch


def test_timeout_unit_conversion() -> None:
    assert run_fetch(timeout_ms=50) == "ok"
