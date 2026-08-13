import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE
    cls_token = encoder_output[:, 0, :]
    norm_cls = (cls_token - np.mean(cls_token, axis=-1, keepdims=True)) / (np.std(cls_token, axis=-1, keepdims=True) + 1e-6)
    logits = norm_cls @ W_head
    return logits