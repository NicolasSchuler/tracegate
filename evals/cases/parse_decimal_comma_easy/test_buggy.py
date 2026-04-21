from buggy import parse_amount


def test_decimal_comma_value() -> None:
    assert parse_amount("12,5") == 12.5
