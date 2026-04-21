from buggy import initialize


def test_initialization_completes() -> None:
    result = initialize()
    assert result["initialized"] is True
