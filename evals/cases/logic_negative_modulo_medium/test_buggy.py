from buggy import circular_get, wrap_index


def test_positive_wrap() -> None:
    assert wrap_index(7, 5) == 2


def test_negative_wrap() -> None:
    assert wrap_index(-1, 5) == 4


def test_negative_large_wrap() -> None:
    assert wrap_index(-7, 5) == 3


def test_zero_index() -> None:
    assert wrap_index(0, 5) == 0


def test_exact_length() -> None:
    assert wrap_index(5, 5) == 0


def test_circular_get_negative() -> None:
    buf = [10, 20, 30, 40, 50]
    assert circular_get(buf, -1) == 50


def test_circular_get_overflow() -> None:
    buf = [10, 20, 30, 40, 50]
    assert circular_get(buf, 6) == 20
