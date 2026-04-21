from buggy import read_numeric_key


def test_json_numeric_key_lookup() -> None:
    assert read_numeric_key('{"1": 99}') == 99
