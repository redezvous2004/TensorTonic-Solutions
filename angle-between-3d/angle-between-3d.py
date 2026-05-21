import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v, w = map(lambda a: np.asarray(a), [v, w])
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm == 0 or w_norm == 0:
        return np.nan
    cosin = np.clip((v @ w) / (v_norm * w_norm), -1, 1)
    return np.arccos(cosin)
    