import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last).
        """
        # YOUR CODE HERE
        N, T, _ = X.shape
        hidden_state = np.zeros((N, self.hidden_dim))
        outputs = []
        for t in range(T):
            input = X[:, t, :]
            prev_concat = np.concatenate([hidden_state, input], axis=-1)
            reset_gate = sigmoid(prev_concat @ self.W_r.T + self.b_r)
            update_gate = sigmoid(prev_concat @ self.W_z.T + self.b_z)

            new_prev_hidden = reset_gate * hidden_state
            post_concat = np.concatenate([new_prev_hidden, input], axis=-1)
            candidate_hidden = np.tanh(post_concat @ self.W_h.T + self.b_h)
            
            hidden_state = update_gate * hidden_state + (1 - update_gate) * candidate_hidden
            y_t = hidden_state @ self.W_y.T + self.b_y

            outputs.append(y_t)
        y = np.stack(outputs, axis=1)
        return y, hidden_state