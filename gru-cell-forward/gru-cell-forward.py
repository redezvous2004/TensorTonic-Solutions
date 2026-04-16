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

    H = h_prev_2d.shape[1]

    W_gate = np.concatenate([params["Wz"], params["Wr"]], axis=1)
    U_gate = np.concatenate([params["Uz"], params["Ur"]], axis=1)
    b_gate = np.concatenate([params["bz"], params["br"]])

    gate = x_2d @ W_gate + h_prev_2d @ U_gate + b_gate
    
    z_t = _sigmoid(gate[:, :H])
    r_t = _sigmoid(gate[:, H:])
    cand_hidden_state = np.tanh(x_2d.dot(params["Wh"]) + (r_t * h_prev_2d).dot(params["Uh"]) + params["bh"])
    h_t = (1 - z_t) * h_prev_2d + z_t * cand_hidden_state
    return h_t.reshape(-1) if is_conversion else h_t
    

    