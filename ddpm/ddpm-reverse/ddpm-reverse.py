import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    result = None
    x_t, epsilon_pred, betas = map(lambda a: np.asarray(a, dtype=float), [x_t, epsilon_pred, betas])
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    coef = (1 - alphas[t - 1]) / (1 - alpha_bar[t - 1]) ** 0.5
    mean = (1 / alphas[t - 1] ** 0.5) * (x_t - coef * epsilon_pred)
    if t > 1 and z is not None:
        sigma = betas[t - 1] ** 0.5
        result = mean + sigma * np.asarray(z, dtype=float)
    else:
        result = mean
    return result