import numpy as np

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch, output_dim) with tanh-activated values rounded to 4 decimals
    """
    z, W, b = map(lambda x: np.asarray(x), [z, W, b])
    return np.tanh(z @ W + b)