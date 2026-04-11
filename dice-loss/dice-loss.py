import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p, y = map(lambda x: np.asarray(x, dtype=float), [p, y])
    if p.ndim != 1:
        p = p.flatten()
    if y.ndim != 1:
        y = y.flatten()
    dice = (2 * np.dot(p, y) + eps) / (np.sum(p) + np.sum(y) + eps)
    dice_loss = 1 - dice
    return dice_loss