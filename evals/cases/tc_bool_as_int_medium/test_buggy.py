from buggy import passing_percentage


def test_passing_percentage() -> None:
    scores = [80.0, 55.0, 90.0, 40.0]
    assert passing_percentage(scores) == "50.0%"
