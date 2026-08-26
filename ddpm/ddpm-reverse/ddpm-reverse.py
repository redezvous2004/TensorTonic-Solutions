import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    x_t, epsilon_pred, betas = map(lambda a: np.asarray(a, dtype=float), [x_t, epsilon_pred, betas])
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    coef = (1 - alphas[t - 1]) / (1 - alpha_bar[t - 1]) ** 0.5
    mean = (1 / alphas[t - 1] ** 0.5) * (x_t - coef * epsilon_pred)
    if t == 1:
        return mean
    if z is None:
        z = np.random.rand(*x_t.shape)
    else:
        z = np.asarray(z, dtype=float)
    return mean + betas[t - 1] ** 0.5 * z