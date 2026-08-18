import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    s = np.linalg.norm(W_hh, ord=2)
    norms = []
    cur_norm = 1.0
    for t in range(T):
        norms.append(cur_norm)
        cur_norm *= s
    return norms