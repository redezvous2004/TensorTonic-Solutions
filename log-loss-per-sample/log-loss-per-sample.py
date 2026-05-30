import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    log_losses = []
    for p_true, p_pred in zip(y_true, y_pred):
        p_pred = min(max(eps, p_pred), 1- eps)
        log_loss = -(p_true * math.log(p_pred) + (1 - p_true) * math.log(1 - p_pred))
        log_losses.append(log_loss)
    return log_losses