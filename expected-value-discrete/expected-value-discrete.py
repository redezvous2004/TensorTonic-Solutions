import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x, p = map(lambda a: np.array(a, dtype=float), [x, p])
    if np.sum(p) < 1 - 1e-6 or np.sum(p) > 1 + 1e-6:
        raise ValueError("Value error!")
    assert x.shape == p.shape, 'Shape mismatch'
    return np.sum(x * p)
