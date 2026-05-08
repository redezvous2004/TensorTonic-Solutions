import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    if strategy == 'mean':
        val_cols = np.nanmean(X, axis=0)
    else:
        val_cols = np.nanmedian(X, axis=0)
    val_cols = np.nan_to_num(val_cols, nan=0.0)
    X_copy = X.copy()
    mask = np.isnan(X_copy)
    if X_copy.ndim == 1:
        X_copy[mask] = val_cols
    else:
        m = X.shape[1]
        for j in range(m):
            X_copy[mask[:,j], j] = val_cols[j]
    return X_copy