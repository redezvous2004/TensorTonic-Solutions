def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code 
    histogram = [0 for _ in range(256)]
    for i in range(len(image)):
        for j in range(len(image[0])):
            histogram[image[i][j]] += 1
    return histogram