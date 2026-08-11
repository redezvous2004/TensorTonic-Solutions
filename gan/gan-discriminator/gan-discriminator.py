import numpy as np

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    x, W = map(lambda a: np.asarray(a), [x, W])
    return 1 / (1 + np.exp(-x @ W))