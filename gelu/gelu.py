import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    vectorized = np.vectorize(math.erf)
    GELU_x = (1 + vectorized(x / np.sqrt(2))) * x * 0.5
    return GELU_x
