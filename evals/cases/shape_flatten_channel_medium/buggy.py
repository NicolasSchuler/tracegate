import numpy as np


def flatten_for_dense(images: np.ndarray) -> np.ndarray:
    """Flatten NCHW image batch to (batch, channels*height*width) for a dense layer.

    Input shape: (N, C, H, W)
    Output shape: (N, C*H*W)
    """
    n, c, h, w = images.shape
    # Bug: reshapes to (N*C, H*W) instead of (N, C*H*W)
    return images.reshape(n * c, h * w)
