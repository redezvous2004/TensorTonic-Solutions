import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix, dtype=float)
    except:
        return None
    if matrix.ndim < 2 or matrix.size == 0:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    eigenvalues, _ = np.linalg.eig(matrix)
    return np.asarray(eigenvalues)