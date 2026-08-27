import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    result = None
    x_t, epsilon_pred, betas = map(lambda a: np.asarray(a, dtype=float), [x_t, epsilon_pred, betas])
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)[t - 1]
    sigma = betas[t - 1] ** 0.5

    coef1 = 1 / alphas[t - 1] ** 0.5
    coef2 = (1 - alphas[t - 1]) / (1 - alpha_bar) ** 0.5

    mean = coef1 * (x_t - coef2 * epsilon_pred)
    if t > 1 and z is not None:
        result = mean + sigma * np.asarray(z, dtype=float)
    else:
        result = mean
    return result
    