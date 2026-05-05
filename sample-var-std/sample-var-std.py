import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    sample_var = np.var(x, ddof=1)
    std = np.sqrt(sample_var)
    return sample_var, std