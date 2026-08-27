import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    return np.round(np.linspace(beta_1, beta_T, T), 6)

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    steps = np.arange(1, T + 1)
    f_t = np.cos((steps / T + s) / (1 + s) * (np.pi / 2)) ** 2
    f_0 = np.cos(s / (1 + s) * (np.pi / 2)) ** 2

    alpha_bar = f_t / f_0
    alpha_bar = np.clip(alpha_bar, 0.0001, 0.9999)
    return np.round(alpha_bar, 6)
    
def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    alpha_bars = np.asarray(alpha_bars, dtype=float)
    alpha_bars_prev = np.append(1.0, alpha_bars[:-1])

    betas = 1 - (alpha_bars / alpha_bars_prev)
    betas = np.clip(betas, 0.0001, 0.9999)
    return np.round(betas, 6)