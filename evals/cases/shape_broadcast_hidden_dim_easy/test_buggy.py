import numpy as np
from buggy import add_bias


def test_bias_broadcast_shape() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    bias = np.array([1.0, -1.0, 0.5])
    result = add_bias(matrix, bias)
    assert result.shape == (2, 3)
