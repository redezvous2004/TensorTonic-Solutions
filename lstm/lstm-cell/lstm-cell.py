import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    # YOUR CODE HERE
    concat = np.concatenate([h_prev, x_t], axis=-1)
    forget_gate = sigmoid(concat @ W_f.T + b_f)
    input_gate = sigmoid(concat @ W_i.T + b_i)
    candidate = np.tanh(concat @ W_c.T + b_c)
    output_gate = sigmoid(concat @ W_o.T + b_o)

    updated_cell = forget_gate * C_prev + input_gate * candidate
    cur_hidden_state = output_gate * np.tanh(updated_cell)
    return cur_hidden_state, updated_cell