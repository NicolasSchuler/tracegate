from buggy import parse_timestamp


def test_datetime_timezone_parse() -> None:
    parsed = parse_timestamp("2024-01-01T12:30:00+00:00")
    assert parsed.isoformat() == "2024-01-01T12:30:00+00:00"
