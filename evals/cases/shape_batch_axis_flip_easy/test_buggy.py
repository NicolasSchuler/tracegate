import numpy as np
from buggy import merge_batches


def test_batch_axis_flip() -> None:
    left = np.ones((2, 3))
    right = np.ones((4, 3))
    merged = merge_batches(left, right)
    assert merged.shape == (6, 3)
