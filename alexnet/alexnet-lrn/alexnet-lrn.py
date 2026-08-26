import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels.
    """
    # YOUR CODE HERE
    b, h, w, c = x.shape
    squared = x ** 2
    out = np.zeros_like(x)

    for i in range(c):
        start = max(0, i - n // 2)
        end = min(c, i + n // 2 + 1)
        channel_sum = np.sum(squared[:, :, :, start:end], axis=-1)
        out[:, :, :, i] = x[:, :, :, i] / (k + alpha * channel_sum) ** beta
    return out