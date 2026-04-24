import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a, b, y = map(lambda x: np.asarray(x, dtype=float), [a, b, y])
    distances = np.linalg.norm(a - b, axis=-1)
    loss = y * (distances ** 2) + (1 - y) * (np.maximum(0, margin - distances) ** 2)
    if reduction == "mean":
        return np.mean(loss)
    else:
        return np.sum(loss)