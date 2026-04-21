import numpy as np


def first_row_mean(batch: np.ndarray) -> float:
    return float(batch[0].mean())
