import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    norm_x = (X - np.min(X, axis=axis, keepdims=True)) / (np.max(X, axis=axis, keepdims=True) - np.min(X, axis=axis, keepdims=True) + eps)
    return norm_x