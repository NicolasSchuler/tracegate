import numpy as np
from buggy import project_features


def test_project_features_shape() -> None:
    X = np.random.randn(8, 4)  # 8 samples, 4 features
    W = np.random.randn(4, 3)  # project 4 features -> 3 outputs
    result = project_features(X, W)
    assert result.shape == (8, 3)


def test_project_features_values() -> None:
    X = np.eye(3)
    W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = project_features(X, W)
    np.testing.assert_array_equal(result, W)
