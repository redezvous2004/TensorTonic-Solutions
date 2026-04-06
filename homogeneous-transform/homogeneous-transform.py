import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T, points = map(lambda x: np.asarray(x, dtype=float), [T, points])
    is_single_point = points.ndim == 1
    if is_single_point:
        points = points.reshape(1, -1)
    ones = np.ones((points.shape[0], 1))
    points = np.hstack([points, ones])

    result_h = points @ T.T
    result = result_h[:, :3]
    return result[0] if is_single_point else result
    