import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.asarray(x)
    pmf = np.where(x == 0, 1 - p, p)
    mean = p
    variance = p * (1 - p)
    return pmf, mean, variance