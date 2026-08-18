import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    h_list = []
    batch, T, input_dim = X.shape
    h = h_0
    for t in range(T):
        input = X[:, t, :]
        h = np.tanh(input @ W_xh.T + h @ W_hh.T + b_h)
        h_list.append(h)
    hidden_states = np.stack(h_list, axis=1)
    h_final = hidden_states[:, -1, :]
    return hidden_states, h_final

    
    