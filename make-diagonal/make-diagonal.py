import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v = np.asarray(v)
    n = len(v)
    diag_matrix = np.zeros((n, n))
    diag_idx = np.diag_indices_from(diag_matrix)
    diag_matrix[diag_idx] = v
    return diag_matrix