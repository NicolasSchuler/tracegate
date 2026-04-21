from buggy import get_results


def test_completed_tasks_returned() -> None:
    result = get_results()
    assert sorted(result) == ["fast", "quick"]
