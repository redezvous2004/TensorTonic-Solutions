import numpy as np
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    identity_matrix = [[1 if i == j else 0 for j in range(len(X[0]))] for i in range(len(X[0]))]
    X, y, identity_matrix = map(lambda a: np.asarray(a), [X, y, identity_matrix])
    weights = np.linalg.inv(X.T @ X + lam * identity_matrix) @ X.T @ y
    return weights