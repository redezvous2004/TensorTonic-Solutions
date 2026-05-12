import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train, X_test = map(lambda x: np.asarray(x), [X_train, X_test])
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    X_test = X_test[:, np.newaxis, :]
    X_train = X_train[np.newaxis, :, :]
    distances = np.linalg.norm(X_test - X_train, axis=-1)
    sorted_idx = np.argsort(distances, axis=-1)
    top_k = np.full((len(X_test), k), -1)
    actual_k = min(k, sorted_idx.shape[1])
    top_k[:, :actual_k] = sorted_idx[:, :actual_k]
    return top_k
    
    