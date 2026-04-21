from buggy import run


def test_greeting_repeated() -> None:
    assert run() == "Hello, World! Hello, World!"
