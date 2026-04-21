import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    mean_val = np.mean(X, axis=axis, keepdims=True)
    std_val = np.std(X, axis=axis, keepdims=True)
    return (X - mean_val) / (std_val + eps)
    