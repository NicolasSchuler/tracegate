from buggy import compute_average


def test_average_is_float() -> None:
    assert compute_average([1, 2, 3, 4]) == 2.5
