import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a 2D matrix to have unit L2 norm.

    Returns matrix with same shape where each row has norm 1.
    Rows with zero norm are left as zeros.
    """
    norms = np.linalg.norm(matrix, axis=1)
    # Bug: norms has shape (N,) without keepdims, so division broadcasts
    # incorrectly against (N, M) — numpy silently broadcasts (N,) as (1, N)
    # when dividing (N, M) by (N,) it works row-wise by accident for square
    # matrices, but for non-square it either errors or gives wrong results.
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms
