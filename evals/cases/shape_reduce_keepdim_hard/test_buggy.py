import numpy as np
from buggy import normalize_rows


def test_normalize_rows_nonsquare() -> None:
    m = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    result = normalize_rows(m)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result[1], [0.0, 0.6, 0.8])


def test_normalize_rows_square() -> None:
    """Square matrix gives wrong values -- the bug is subtle here."""
    m = np.array([[3.0, 4.0], [0.0, 0.0]])
    result = normalize_rows(m)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0)
    np.testing.assert_array_equal(result[1], [0.0, 0.0])


def test_normalize_rows_single_row() -> None:
    m = np.array([[5.0, 0.0]])
    result = normalize_rows(m)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result[0], [1.0, 0.0])
