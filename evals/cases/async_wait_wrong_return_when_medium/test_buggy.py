from buggy import run


def test_first_result_returned_quickly() -> None:
    assert run() == "fast_result"
