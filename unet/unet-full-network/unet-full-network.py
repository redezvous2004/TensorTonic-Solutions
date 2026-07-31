import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    b, h, w, c = x.shape
    e1h, e1w = (h - 4) // 2, (w - 4) // 2
    e2h, e2w = (e1h - 4) // 2, (e1w - 4) // 2
    e3h, e3w = (e2h - 4) // 2, (e2w - 4) // 2
    e4h, e4w = (e3h - 4) // 2, (e3w - 4) // 2
    bh, bw = e4h - 4, e4w - 4
    d1h, d1w = bh * 2 - 4, bw * 2 - 4
    d2h, d2w = d1h * 2 - 4, d1w * 2 - 4
    d3h, d3w = d2h * 2 - 4, d2w * 2 - 4
    d4h, d4w = d3h * 2 - 4, d3w * 2 - 4
    return np.zeros((b, d4h, d4w, num_classes))
    
