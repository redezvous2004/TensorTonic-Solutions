import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x)
    if rng is None:
        idx_arr = np.random.randint(len(x), size=(n_bootstrap, len(x)))
    else:
        idx_arr = rng.integers(len(x), size=(n_bootstrap, len(x)))
    bootstrap_samples = x[idx_arr]
    boot_means = np.mean(bootstrap_samples, axis=-1)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return boot_means, lower, upper
