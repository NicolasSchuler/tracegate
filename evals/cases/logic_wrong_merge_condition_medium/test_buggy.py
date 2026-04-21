from buggy import merge_sort


def test_sort_basic() -> None:
    assert merge_sort([3, 1, 2]) == [1, 2, 3]


def test_sort_already_sorted() -> None:
    assert merge_sort([1, 2, 3, 4]) == [1, 2, 3, 4]


def test_sort_reverse() -> None:
    assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_sort_duplicates() -> None:
    assert merge_sort([3, 1, 3, 2, 1]) == [1, 1, 2, 3, 3]


def test_sort_single() -> None:
    assert merge_sort([42]) == [42]


def test_sort_empty() -> None:
    assert merge_sort([]) == []
