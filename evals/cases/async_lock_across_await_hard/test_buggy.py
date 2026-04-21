from buggy import run


def test_concurrent_increments_are_atomic() -> None:
    assert run() == 3
