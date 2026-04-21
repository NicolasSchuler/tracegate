import numpy as np
from buggy import flatten_for_dense


def test_flatten_shape() -> None:
    images = np.random.randn(4, 3, 8, 8)  # 4 images, 3 channels, 8x8
    flat = flatten_for_dense(images)
    assert flat.shape == (4, 192)


def test_flatten_preserves_batch() -> None:
    images = np.zeros((2, 1, 2, 2))
    images[0] = 1.0
    images[1] = 2.0
    flat = flatten_for_dense(images)
    assert flat.shape[0] == 2
    np.testing.assert_array_equal(flat[0], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(flat[1], [2.0, 2.0, 2.0, 2.0])
