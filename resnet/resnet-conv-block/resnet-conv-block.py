import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    x, W1, W2, Ws = map(lambda a: np.asarray(a), [x, W1, W2, Ws])
    h = np.maximum(0, x @ W1)
    z = h @ W2
    s = x @ Ws
    return np.maximum(0, z + s)
