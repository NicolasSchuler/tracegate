from buggy import check_tax_balance


def test_tax_balance_check() -> None:
    prices = [0.1, 0.2, 0.3]
    # Tax at 10%: 0.01 + 0.02 + 0.03 = 0.06
    assert check_tax_balance(prices, 0.06)
