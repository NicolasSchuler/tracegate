from buggy import read_user_id


def test_field_rename_contract() -> None:
    assert read_user_id(5) == 5
