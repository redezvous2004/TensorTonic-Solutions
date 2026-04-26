import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim != 2 or len(X) < 2:
        return None
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    std =np.std(X_centered, axis=0, ddof=1)
    covariance_mtx = (1 / (len(X) - 1)) * (X_centered.T @ X_centered)
    return covariance_mtx / np.outer(std, std)
    