import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    x_t, h_prev, h_t, W, U, b = cache
    dh, x_t, h_prev, h_t, W, U, b = map(lambda a: np.asarray(a), [dh, x_t, h_prev, h_t, W, U, b])
    dtanh = dh * (1 - h_t ** 2)
    dx_t = W.T @ dtanh
    dh_prev = U.T @ dtanh
    dW = np.outer(dtanh, x_t)
    dU = np.outer(dtanh, h_prev)
    db = dtanh
    return dx_t, dh_prev, dW, dU, db