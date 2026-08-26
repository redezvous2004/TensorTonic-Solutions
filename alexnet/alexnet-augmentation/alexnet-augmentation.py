import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224, crop_y: int = None, crop_x: int = None) -> np.ndarray:
    """
    Extract a crop from the image at (crop_y, crop_x). If not given, choose randomly.
    """
    # YOUR CODE HERE
    h, w, c = image.shape
    if crop_y is None:
        crop_y = np.random.randint(0, h - crop_size)
    if crop_x is None:
        crop_x = np.random.randint(0, w - crop_size)

    return image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5, flip_rand: float = None) -> np.ndarray:
    """
    Flip image horizontally if flip_rand < p. If flip_rand not given, generate randomly.
    """
    # YOUR CODE HERE
    if flip_rand is None:
        flip_rand = np.random.rand()
    if flip_rand < p:
        return image[:, ::-1, :]
    else:
        return image