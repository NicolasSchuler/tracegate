from buggy import get_all


def test_gather_returns_errors_with_results() -> None:
    urls = ["https://good.com", "https://bad.com", "https://also-good.com"]
    results = get_all(urls)
    assert len(results) == 3
    assert results[0] == "data from https://good.com"
    assert isinstance(results[1], ValueError)
    assert results[2] == "data from https://also-good.com"
