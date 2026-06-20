def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    height, width = len(image), len(image[0])
    gray_scale_image = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            gray_scale_image[i][j] = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]
    return gray_scale_image
                
                
    