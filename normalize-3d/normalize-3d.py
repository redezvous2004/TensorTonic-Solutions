import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    v_length = np.linalg.norm(v, axis=-1, keepdims=True)
    v_normed = np.divide(v, v_length, out=np.zeros_like(v), where=v_length != 0)
    return v_normed