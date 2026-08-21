import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate and hidden state."""
    # YOUR CODE HERE
    concat = np.concatenate([h_prev, x_t], axis=-1)
    out = sigmoid(concat @ W_o.T + b_o)
    new_hstate = out * np.tanh(C_t)
    return out, new_hstate