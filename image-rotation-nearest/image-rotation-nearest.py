import math
def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    # Write code here
    h, w = len(image), len(image[0])
    angle_rad = angle_degrees * (math.pi / 180)
    output = [[0] * w for _ in range(h)]
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    for i in range(h):
        for j in range(w):
            dy = i - cy
            dx = j - cx
            src_y = round(cy + dy * math.cos(angle_rad) + dx * math.sin(angle_rad))
            src_x = round(cx - dy * math.sin(angle_rad) + dx * math.cos(angle_rad))
            if src_y < 0 or src_y > h - 1 or src_x < 0 or src_x > w - 1:
                output[i][j] = 0
            else:
                output[i][j] = image[src_y][src_x]
    return output
            
    