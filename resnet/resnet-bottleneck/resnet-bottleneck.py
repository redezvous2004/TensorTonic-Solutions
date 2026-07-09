import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    x, W1, W2, W3, Ws = map(lambda a: np.asarray(a), [x, W1, W2, W3, Ws])
    x1 = np.maximum(0, x @ W1)
    x2 = np.maximum(0, x1 @ W2)
    x3 = x2 @ W3
    out = np.maximum(0, x3 + x @ Ws)
    return out
