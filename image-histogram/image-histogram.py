import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code 
    image = np.asarray(image)
    image_flatten = image.flatten()
    return np.bincount(image_flatten, minlength=256).tolist()