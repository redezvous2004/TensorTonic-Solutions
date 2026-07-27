import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    b, h, w, c = x.shape
    x = x.reshape(b, h // 2, 2, w // 2, 2, c)
    result = np.max(x, axis=(2, 4))
    return result
    
    