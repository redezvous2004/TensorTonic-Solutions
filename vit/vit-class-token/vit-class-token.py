import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    b, n, _ = patches.shape
    if cls_token is None:
        cls_token = np.random.randn(1, 1, embed_dim) * 0.02

    cls = np.tile(cls_token, (b, 1, 1))
    out = np.concatenate((cls, patches), axis=1)
    return out