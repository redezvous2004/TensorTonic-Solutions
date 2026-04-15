import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true, y_pred = map(lambda x: np.asarray(x), [y_true, y_pred])
    N = len(y_true)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    probs = y_pred[np.arange(N), y_true]

    cross_entropy_loss = -np.mean(np.log(probs))
    return cross_entropy_loss