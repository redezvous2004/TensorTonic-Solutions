import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a, b = map(lambda x: np.asarray(x), [a, b])
    len_vec_a, len_vec_b = np.linalg.norm(a), np.linalg.norm(b)
    if len_vec_a == 0 or len_vec_b == 0:
        return 0
    else:
        return (a @ b) / (len_vec_a * len_vec_b)