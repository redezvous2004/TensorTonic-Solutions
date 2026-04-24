import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or norm_type not in ['l1', 'l2', 'max']:
        return None
    if axis is not None and axis >= matrix.ndim:
        return None
    if norm_type == 'l2':
        norm_vals = np.linalg.norm(matrix, axis=axis, keepdims=True)
    elif norm_type == 'l1':
        norm_vals = np.linalg.norm(matrix, ord=1, axis=axis, keepdims=True)
    else:
        norm_vals = np.linalg.norm(matrix, ord=np.inf, axis=axis, keepdims=True)

    norm_vals[norm_vals == 0] = 1e-6
    return matrix / norm_vals