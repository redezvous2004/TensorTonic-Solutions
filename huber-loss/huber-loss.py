import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true, y_pred = map(lambda x: np.asarray(x), [y_true, y_pred])
    if y_true.ndim != y_pred.ndim:
        return None
    errors = np.abs(y_true - y_pred)
    losses = np.where(errors <= delta, 0.5 * errors ** 2, delta * (errors - 0.5 * delta))
    return np.mean(losses)