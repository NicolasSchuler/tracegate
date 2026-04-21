from buggy import extract_city

PAYLOAD = '{"user": {"name": "Alice", "address": {"city": "Paris", "zip": "75001"}}}'


def test_extract_nested_city() -> None:
    assert extract_city(PAYLOAD) == "Paris"
