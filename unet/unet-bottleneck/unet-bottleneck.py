import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    Two 3x3 unpadded convolutions, no pooling.
    Returns zero array with correct shape.
    """
    # Your implementation here
    b, h, w, c = x.shape
    return np.zeros((b, h - 4, w - 4, out_channels))
