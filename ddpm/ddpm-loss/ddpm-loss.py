import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    # YOUR CODE HERE
    epsilon, epsilon_pred = map(lambda a: np.asarray(a, dtype=float), [epsilon, epsilon_pred])
    loss = np.mean((epsilon - epsilon_pred) ** 2)
    return loss