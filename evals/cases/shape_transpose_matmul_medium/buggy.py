import numpy as np


def project_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project feature matrix X (batch, features) through weight matrix W (features, out).

    Returns (batch, out) matrix.
    """
    # Bug: transposes axis 0 and 1 of X instead of W when shapes mismatch
    return X.T @ W
