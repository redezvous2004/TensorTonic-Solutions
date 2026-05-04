import numpy as np
def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    if lam == 0:
        pmf = 1.0 if k == 0 else 0.0
        cdf = 1.0
        return float(pmf), float(cdf)

    log_factorial = np.sum(np.log(np.arange(1, k + 1)))
    log_pmf = -lam + k * np.log(lam) - log_factorial
    pmf = np.exp(log_pmf)

    log_pmf_array = []
    for i in range(k + 1):
        log_factorial_i = np.sum(np.log(np.arange(1, i + 1)))
        log_pmf_array.append(-lam + i * np.log(lam) - log_factorial_i)
    cdf = np.sum(np.exp(log_pmf_array))
    return float(pmf), float(cdf)