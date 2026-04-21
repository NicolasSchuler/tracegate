from buggy import run


def test_no_concurrent_violations() -> None:
    violations = run()
    assert violations == [], f"concurrent access detected: {violations}"
