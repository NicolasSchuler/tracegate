from buggy import upper_user_name


def test_optional_none_contract() -> None:
    assert upper_user_name(False) == ""
