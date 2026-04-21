from buggy import run


def test_condition_pipeline() -> None:
    assert run() == 42
