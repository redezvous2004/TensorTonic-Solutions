import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    t_stat = (mean - mu0) / (std / np.sqrt(len(x)))
    return t_stat