import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.asarray(scores)
    seq_len = scores.shape[-1]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores_copy = scores.copy()
    scores_copy[..., mask] = mask_value
    return scores_copy