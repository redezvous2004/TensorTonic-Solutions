import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    real_data, fake_data, D_W = map(lambda a: np.asarray(a), [real_data, fake_data, D_W])
    p_real = 1 / (1 + np.exp(-real_data @ D_W))
    p_fake = 1 / (1 + np.exp(-fake_data @ D_W))
    p_real, p_fake = map(lambda x: np.clip(x, 1e-8, 1 - 1e-8), [p_real, p_fake])

    d_loss = -np.mean(np.log(p_real) + np.log(1 - p_fake))
    g_loss = -np.mean(np.log(p_fake))
    return {
        "d_loss": d_loss,
        "g_loss": g_loss
    }