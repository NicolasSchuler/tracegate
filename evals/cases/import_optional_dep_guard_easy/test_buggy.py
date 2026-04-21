from buggy import parse_payload


def test_optional_dependency_guard() -> None:
    assert parse_payload('{"value": 3}') == {"value": 3}
