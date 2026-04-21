from buggy import fetch_all_pages


def test_both_passes_have_items() -> None:
    result = fetch_all_pages(3)
    assert result == [0, 10, 20, 0, 10, 20]
