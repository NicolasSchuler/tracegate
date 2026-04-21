from buggy import last_item


def test_index_control_case() -> None:
    assert last_item([5, 6, 7]) == 7
