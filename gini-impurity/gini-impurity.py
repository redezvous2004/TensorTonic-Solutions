import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left, y_right = map(lambda x: np.asarray(x), [y_left, y_right])
    Nl, Nr, N = len(y_left), len(y_right), len(y_left) + len(y_right)
    
    _, counts_left = np.unique(y_left, return_counts=True)
    _, counts_right = np.unique(y_right, return_counts=True)

    gini_left = 1 - np.sum(np.pow(counts_left / Nl, 2)) if Nl != 0 else 0.0
    gini_right = 1 - np.sum(np.pow(counts_right / Nr, 2)) if Nr != 0 else 0.0

    gini_split = (Nl / N) * gini_left + (Nr / N) * gini_right if N != 0 else 0.0
    return gini_split