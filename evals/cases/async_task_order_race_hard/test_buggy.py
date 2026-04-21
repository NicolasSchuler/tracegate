from buggy import racey_read


def test_task_is_awaited_before_result_read() -> None:
    assert racey_read() == 8
