import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g, dtype=float)
    glob_grad = np.sqrt(np.sum(g * g))
    if glob_grad == 0 or max_norm <= 0:
        return g
    if glob_grad > max_norm:
        return g * (max_norm / glob_grad)
    return g