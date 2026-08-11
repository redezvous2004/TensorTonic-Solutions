import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    real_probs, fake_probs = map(lambda x: np.clip(np.asarray(x), 1e-8, 1 - 1e-8), [real_probs, fake_probs])
    loss_d = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    return loss_d
def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    fake_probs = np.clip(np.asarray(fake_probs), 1e-8, None)
    loss_g = -np.mean(np.log(fake_probs))
    return loss_g
