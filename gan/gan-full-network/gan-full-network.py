import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        z = np.array(z)
        return np.tanh(z @ self.G_W)
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        x = np.array(x)
        return 1 / (1 + np.exp(-x @ self.D_W))
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        # Your implementation here
        generated_data = self.generate(z)
        p_fake = self.discriminate(generated_data)
        p_real = self.discriminate(real_data)
        p_fake, p_real = map(lambda a: np.clip(a, 1e-8, 1 - 1e-8), [p_fake, p_real])
        
        d_loss = -np.mean(np.log(p_real) + np.log(1 - p_fake))
        g_loss = -np.mean(np.log(p_fake))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss
        }
        
        