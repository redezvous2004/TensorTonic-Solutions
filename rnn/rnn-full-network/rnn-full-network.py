import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        batch, T, input_dim = X.shape
        if h_0 is None:
            h_0 = np.zeros(batch, input_dim)
        h = h_0
        projected_hidden = []
        for t in range(T):
            input = X[:, t, :]
            h = np.tanh(input @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            projected_hidden.append(h @ self.W_hy.T + self.b_y)
        y_seq = np.stack(projected_hidden, axis=1)
        return y_seq, h