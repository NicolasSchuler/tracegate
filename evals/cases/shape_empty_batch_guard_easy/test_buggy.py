import numpy as np
from buggy import first_row_mean


def test_empty_batch_guard() -> None:
    batch = np.empty((0, 3))
    assert first_row_mean(batch) == 0.0
