from buggy import persist


def test_save_completes() -> None:
    assert persist("hello") == "saved:hello"
