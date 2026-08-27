import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    """
    Returns: np.ndarray of the final denoised sample
    """
    # YOUR CODE HERE
    x_T, betas, epsilon_preds = map(lambda a: np.asarray(a, dtype=float), [x_T, betas, epsilon_preds])
    T = len(betas)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    cur_x = x_T.copy()
    for i, t in enumerate(range(T, 0, -1)):
        alphas_t = alphas[t - 1]
        betas_t = betas[t - 1]
        alpha_bar_t = alpha_bar[t - 1]

        coef1 = 1 / alphas_t ** 0.5
        coef2 = (1 - alphas_t) / (1 - alpha_bar_t) ** 0.5
        mean = coef1 * (cur_x - coef2 * epsilon_preds[i])

        if t > 1:
            sigma = betas[t - 1] ** 0.5
            cur_x = mean + sigma * np.asarray(z_values[i], dtype=float)
        else:
            cur_x = mean
    return cur_x