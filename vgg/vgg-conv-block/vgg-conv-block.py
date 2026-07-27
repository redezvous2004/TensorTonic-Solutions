import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    # Your implementation here
    out= x.copy()
    for w, b in zip(weights, biases):
        w, b = map(lambda x: np.array(x), [w, b])
        out = np.maximum(0, out @ w + b)
    return out