def accumulate_tax(prices: list[float], rate: float = 0.1) -> float:
    """Sum tax amounts for each price and return the total."""
    total = 0.0
    for price in prices:
        total += price * rate
    return total


def check_tax_balance(prices: list[float], expected: float) -> bool:
    """Return True if computed tax matches expected value exactly."""
    computed = accumulate_tax(prices)
    return computed == expected
