from buggy import compute_total


def test_scores_summed() -> None:
    assert compute_total([1, 2, 3]) == 60
