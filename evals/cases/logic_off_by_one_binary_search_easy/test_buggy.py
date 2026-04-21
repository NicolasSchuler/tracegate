from buggy import binary_search


def test_find_existing() -> None:
    arr = [1, 3, 5, 7, 9]
    assert binary_search(arr, 5) == 2


def test_find_first() -> None:
    arr = [1, 3, 5, 7, 9]
    assert binary_search(arr, 1) == 0


def test_find_last() -> None:
    arr = [1, 3, 5, 7, 9]
    assert binary_search(arr, 9) == 4


def test_not_found() -> None:
    arr = [1, 3, 5, 7, 9]
    assert binary_search(arr, 6) == -1


def test_empty() -> None:
    assert binary_search([], 1) == -1
