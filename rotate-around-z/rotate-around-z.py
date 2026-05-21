import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    # Your code here
    points = np.asarray(points)
    shape = points.shape
    if points.ndim == 1:
        points = points.reshape(1, -1)
    rotation_matrix = np.asarray([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    transformed_matrix = points @ rotation_matrix.T
    return transformed_matrix.reshape(shape)