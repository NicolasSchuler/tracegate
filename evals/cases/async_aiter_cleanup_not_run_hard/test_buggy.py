from buggy import get_first


def test_cleanup_after_early_break() -> None:
    items, cleaned = get_first(10, 3)
    assert items == [0, 10, 20]
    assert cleaned, "generator cleanup did not run after early break"
