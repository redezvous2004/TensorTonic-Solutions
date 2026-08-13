import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    b, h, w, c = image.shape
    n = (h // patch_size) * (w // patch_size)
    patch_dim = patch_size * patch_size * c
    reshaped_img = image.reshape(b, h // patch_size, patch_size, w // patch_size, patch_size, c)
    reshaped_img = reshaped_img.transpose(0, 1, 3, 2, 4, 5)
    reshaped_img = reshaped_img.reshape(b, n, patch_dim)
    embeddings = reshaped_img @ W_proj
    return embeddings