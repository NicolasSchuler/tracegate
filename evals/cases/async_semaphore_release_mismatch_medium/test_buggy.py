from buggy import run_fetch


def test_fetch_with_errors() -> None:
    urls = ["ok1", "bad1", "ok2"]
    result = run_fetch(urls)
    assert result == ["data:ok1", "error", "data:ok2"]
