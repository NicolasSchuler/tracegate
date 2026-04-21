from buggy import BatchNormalizer


def test_previous_batch_scale_does_not_leak() -> None:
    normalizer = BatchNormalizer()
    assert normalizer.normalize([2, 4]) == [0.5, 1.0]
    assert normalizer.normalize([1, 3]) == [1 / 3, 1.0]
