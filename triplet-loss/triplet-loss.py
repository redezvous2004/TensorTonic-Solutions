import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor, positive, negative = map(lambda x: np.asarray(x), [anchor, positive, negative])
    losses = np.maximum(0, np.linalg.norm(anchor - positive, axis=-1) ** 2  - np.linalg.norm(anchor - negative, axis=-1) ** 2 + margin)
    return np.mean(losses)