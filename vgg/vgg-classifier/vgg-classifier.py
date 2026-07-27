import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    # Your implementation here
    b, h, w, c = features.shape
    flatted_features = features.reshape(b, -1)
    y1 = np.maximum(0, flatted_features @ W1 + b1)
    y2 = np.maximum(0, y1 @ W2 + b2)
    y3 = y2 @ W3 + b3
    return y3
    