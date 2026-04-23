import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x, gamma, beta = map(lambda a: np.asarray(a, dtype=float), [x, gamma, beta])
    if x.ndim == 2:
        calc_axis = 0
        shape = (1, -1)
    else:
        calc_axis = (0, 2, 3)
        shape = (1, -1, 1, 1)

    mean_val = np.mean(x, axis=calc_axis, keepdims=True)
    var_val = np.var(x, axis=calc_axis, keepdims=True)
    x_hat = (x - mean_val) / np.sqrt(var_val + eps)
    y = gamma.reshape(shape) * x_hat + beta.reshape(shape)
    return y