import math
def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    # Write code here
    if size == 1:
        return [[1.0]]
    center = size // 2
    gaussian_kernel = [[0.0] * size for _ in range(size)]
    kernel_1d = []
    for i in range(size):
        offset = i - center
        val = math.exp(-(offset ** 2) / (2 * sigma ** 2))
        kernel_1d.append(val)
        
    gaussian_kernel = [[0.0] * size for _ in range(size)]
    total = 0.0
    
    for i in range(size):
        for j in range(size):
            gaussian_kernel[i][j] = kernel_1d[i] * kernel_1d[j]
            total += gaussian_kernel[i][j]
            
    for i in range(size):
        for j in range(size):
            gaussian_kernel[i][j] /= total
    return gaussian_kernel
    
            