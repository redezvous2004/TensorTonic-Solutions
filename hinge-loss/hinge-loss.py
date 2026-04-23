import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true, y_score = map(lambda x: np.asarray(x), [y_true, y_score])
    valid = np.all((y_true == 1) | (y_true == -1))
    if y_true.shape != y_score.shape or not valid:
        return 0.0
    loss = np.maximum(0, margin - y_true * y_score)
    if reduction == "mean":
        return np.mean(loss)
    else:
        return np.sum(loss)