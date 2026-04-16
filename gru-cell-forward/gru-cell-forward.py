import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x_copy, h_prev_copy = x.copy(), h_prev.copy()
    x_2d, _ =  _as2d(x_copy, len(x_copy))
    h_prev_2d, is_conversion = _as2d(h_prev_copy, len(h_prev_copy))
    z_t = _sigmoid(x_2d.dot(params["Wz"]) + h_prev_2d.dot(params["Uz"]) + params["bz"])
    r_t = _sigmoid(x_2d.dot(params["Wr"]) + h_prev_2d.dot(params["Ur"]) + params["br"])
    cand_hidden_state = np.tanh(x_2d.dot(params["Wh"]) + (r_t * h_prev_2d).dot(params["Uh"]) + params["bh"])
    h_t = (1 - z_t) * h_prev_2d + z_t * cand_hidden_state
    if is_conversion == True:
        return h_t.reshape(-1)
    else:
        return h_t
    

    