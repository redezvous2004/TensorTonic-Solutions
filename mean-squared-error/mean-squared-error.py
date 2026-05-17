import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_true, y_pred = map(lambda x: np.asarray(x), [y_true, y_pred])
    if y_true.ndim != y_pred.ndim:
        return None

    MSE = np.mean((y_pred - y_true) ** 2)
    return MSE
