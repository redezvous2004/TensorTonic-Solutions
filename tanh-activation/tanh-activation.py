import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    # return np.tanh(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    