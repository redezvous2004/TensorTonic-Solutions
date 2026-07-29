import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Returns zero array with correct shape.
    """
    # Your implementation here
    b, h, w, _ = x.shape
    out = np.zeros((b, 2 * h - 4, 2 * w - 4, out_channels))
    return out
