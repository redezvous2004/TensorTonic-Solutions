import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray(v)
    v_norm = np.linalg.norm(v, axis=-1)
    return v_norm