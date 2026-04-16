import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x, y = map(lambda a: np.asarray(a), [x, y])
    return int(np.sum(np.abs(x - y)))