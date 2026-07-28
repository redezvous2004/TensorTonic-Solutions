import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    # Your implementation here
    out = x.copy()
    i = 0
    for conf in config:
        if conf == 'M':
            out = maxpool_2x2(out)
        else:
            w, b = map(lambda a: np.array(a), [conv_weights[i], conv_biases[i]])
            out = np.maximum(0, out @ w + b)
            i += 1
    return out
    