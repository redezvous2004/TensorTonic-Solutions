import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    # Your implementation here
    b, h, w, c = x.shape
    skip_out_shape = (b, h - 4, w - 4, out_channels)
    skip_out = np.zeros(skip_out_shape, dtype=x.dtype)
    pool_out_shape = (b, skip_out_shape[1] // 2, skip_out_shape[2] // 2, out_channels)
    pool_out = np.zeros(pool_out_shape, dtype=x.dtype)
    return pool_out, skip_out
    
