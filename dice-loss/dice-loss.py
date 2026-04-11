import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p, y = map(lambda x: np.asarray(x, dtype=float), [p, y])

    dice = (2 * np.sum(p * y) + eps) / (np.sum(p) + np.sum(y) + eps)
    dice_loss = 1 - dice
    return dice_loss