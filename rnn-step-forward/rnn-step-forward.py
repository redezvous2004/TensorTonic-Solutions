import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t, h_prev, Wx, Wh, b = map(lambda a: np.asarray(a), [x_t, h_prev, Wx, Wh, b])
    pre_act = x_t @ Wx + h_prev @ Wh + b
    h_t = np.tanh(pre_act)

    return h_t
