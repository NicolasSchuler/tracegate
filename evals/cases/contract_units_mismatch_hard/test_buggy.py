from buggy import timeout_seconds


def test_units_contract() -> None:
    assert timeout_seconds() == 0.25
