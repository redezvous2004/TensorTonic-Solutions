import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p, q = map(lambda x: np.asarray(x), [p, q])
    if np.all(p == q):
        return 0.0
    p, q = p + eps, q + eps
    divergence = np.sum(p * np.log(p / q))
    return divergence