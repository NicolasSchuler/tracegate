from buggy import main


def test_nested_asyncio_run() -> None:
    assert main() == 43
