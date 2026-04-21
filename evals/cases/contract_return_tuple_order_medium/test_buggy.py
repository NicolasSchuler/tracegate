from buggy import read_count


def test_tuple_order_contract() -> None:
    assert read_count() == 2
