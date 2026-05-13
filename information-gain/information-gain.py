import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    parent_entropy = _entropy(y)
    if parent_entropy == 0.0:
        return 0.0
    N = len(split_mask)
    left, right = y[split_mask], y[~split_mask]
    right_child_entropy = _entropy(right)
    left_child_entropy = _entropy(left)
    IG = parent_entropy - (len(left) / N) * left_child_entropy - (len(right) / N) * right_child_entropy
    return IG
    
