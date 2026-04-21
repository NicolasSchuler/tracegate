from buggy import run_jobs


def test_fire_forget_results_collected() -> None:
    result = run_jobs([1, 2, 3])
    assert sorted(result) == [10, 20, 30]
