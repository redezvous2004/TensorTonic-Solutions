import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1, Z2 = map(lambda x: np.asarray(x), [Z1, Z2])
    N = Z1.shape[0]
    similarity_matrix = (Z1 @ Z2.T) / temperature
    max_logits = np.max(similarity_matrix, axis=-1, keepdims=True)
    exp_logits = np.exp(similarity_matrix - max_logits)
    numerator = np.diag(exp_logits)
    denominator = np.sum(exp_logits, axis=-1)
    losses = -np.log(numerator / denominator)
    return np.mean(losses)