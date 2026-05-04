import numpy as np
from scipy.special import gammaln
def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    if lam == 0:
        pmf = 1.0 if k == 0 else 0.0
        cdf = 1.0
        return float(pmf), float(cdf)

    log_pmf = -lam + k * np.log(lam) - gammaln(k + 1)
    pmf = np.exp(log_pmf)

    idx = np.arange(k + 1)
    log_pmf_array = -lam + idx * np.log(lam) - gammaln(idx + 1)
    cdf = np.sum(np.exp(log_pmf_array))
    return float(pmf), float(cdf)