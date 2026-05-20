import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    value_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_normed = np.divide(v, value_norm, out=np.zeros_like(v), where=value_norm != 0)
    return v_normed