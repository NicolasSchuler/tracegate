from buggy import get_batch


def test_batch_fetch() -> None:
    result = get_batch([1, 2, 3, 4, 5])
    assert len(result) == 5
    assert result[0] == {"id": 1, "value": 100}
