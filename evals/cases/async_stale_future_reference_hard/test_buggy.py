from buggy import compute, warm


def test_cached_value_reused() -> None:
    warm("hello")
    assert compute("hello") == 50
