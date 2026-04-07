import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    _, counts = np.unique(y, return_counts=True)
    N = len(y)
    surprise = np.where(counts > 0, counts / N * np.log2(counts / N), 0)
    h = -np.sum(surprise)
    return h