import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    if num_classes is None:
        num_classes = max(y) + 1
    one_hot_mtx = np.zeros((len(y), num_classes))
    one_hot_mtx[np.arange(len(y)), y] = 1
    return one_hot_mtx