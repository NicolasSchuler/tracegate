import numpy as np


def merge_batches(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1)
