import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p, y = map(lambda x: np.asarray(x), [p, y])
    focal_losses = -((1 - p) ** gamma) * y * np.log(p) - p ** gamma * (1 - y) * np.log(1 - p)
    return np.mean(focal_losses)