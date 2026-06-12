import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X, y = map(lambda a: np.asarray(a), [X, y])
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    return weights