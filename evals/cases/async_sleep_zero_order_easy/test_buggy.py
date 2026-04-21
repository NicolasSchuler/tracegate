from buggy import get_ordered_log


def test_items_in_sequence() -> None:
    result = get_ordered_log()
    assert result == ["A-0", "A-1", "A-2", "B-0", "B-1", "B-2"]
