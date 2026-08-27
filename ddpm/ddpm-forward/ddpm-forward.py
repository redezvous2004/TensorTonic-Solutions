import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    betas = np.asarray(betas, dtype=float)
    return np.round(np.cumprod(1.0 - betas), 6)

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    # YOUR CODE HERE
    x_0, epsilon = map(lambda a: np.asarray(a, dtype=float), [x_0, epsilon])
    alpha_bar = get_alpha_bar(betas)
    return  alpha_bar[t - 1] ** 0.5 * x_0 + (1 - alpha_bar[t - 1]) ** 0.5 * epsilon
    