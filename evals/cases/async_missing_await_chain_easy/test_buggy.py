from buggy import compute


def test_missing_await_chain() -> None:
    assert compute(1) == 3
