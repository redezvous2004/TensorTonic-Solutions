import math
def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    # Write code here
    n, m = len(image), len(image[0])
    for pixels in image:
        pixels.insert(0, 0)
        pixels.append(0)
    new_n, new_m = len(image), len(image[0])
    
    image.insert(0, [0] * new_m)
    image.append([0] * new_m)
    
    output = [[0] * m for _ in range(n)]
    k_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    k_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    for i in range(n):
        for j in range(m):
            gx, gy = 0, 0
            for a in range(len(k_x)):
                for b in range(len(k_x[0])):
                    gx += k_x[a][b] * image[i + a][j + b]
                    gy += k_y[a][b] * image[i + a][j + b]
            output[i][j] += math.sqrt(gx ** 2 + gy ** 2)
    return output
            