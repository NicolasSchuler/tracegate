from buggy import format_receipt_line


def test_receipt_line_format() -> None:
    assert format_receipt_line("Coffee", 4.50) == "Coffee: $4.5"
