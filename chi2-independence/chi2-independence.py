import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.asarray(C)
    row_sum = np.sum(C, axis=1)
    column_sum = np.sum(C, axis=0)
    grand_sum = np.sum(C)
    expected_counts = np.zeros(C.shape)
    expected_counts = np.outer(row_sum, column_sum) / grand_sum
    chi2 = np.sum((C - expected_counts) ** 2 / expected_counts)
    return chi2, expected_counts