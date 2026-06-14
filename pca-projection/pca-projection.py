import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X)
    n, _ = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    covariance_matrix = X_centered.T @ X_centered / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_eigenvectors = eigenvectors[:, top_k_indices]
    X_projected = X_centered @ top_k_eigenvectors
    return X_projected