import numpy as np


def add_bias(matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return matrix + bias.reshape(-1, 1)
