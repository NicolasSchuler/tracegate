from buggy import run


def test_external_cancel_not_swallowed() -> None:
    assert run() == "properly_cancelled"
