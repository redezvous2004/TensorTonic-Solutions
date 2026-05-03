import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    real_scores, fake_scores = map(lambda x: np.asarray(x), [real_scores, fake_scores])
    return np.mean(fake_scores) - np.mean(real_scores)