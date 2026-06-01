def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    max_w_len = max(len(row) for row in image)
    for row in image:
        row.extend([0] * (max_w_len - len(row)))
    if padding:
        n = len(image[0])
        for pixels in image:
            for i in range(padding):
                pixels.insert(0, 0)
            pixels.extend([0] * padding)
        new_width = len(image[0])
        for i in range(padding):
            image.insert(0, [0] * new_width)
            image.append([0] * new_width)
    h_image, w_image, h_kernel, w_kernel = len(image), len(image[0]), len(kernel), len(kernel[0])
    h_out = (h_image - h_kernel) // stride + 1
    w_out = (w_image - w_kernel) // stride + 1
    outputs = [[0] * w_out for _ in range(h_out)]
    for i in range(h_out):
        for j in range(w_out):
            for m in range(h_kernel):
                for n in range(w_kernel):
                    outputs[i][j] += image[i * stride + m][j * stride + n] * kernel[m][n]
    return outputs
                    
    