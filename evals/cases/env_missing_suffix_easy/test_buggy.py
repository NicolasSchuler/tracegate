from buggy import load_threshold


def test_missing_suffix() -> None:
    assert load_threshold() == 7
