from buggy import read_cancelled


def test_cancelled_task_is_handled() -> None:
    assert read_cancelled() == 3
