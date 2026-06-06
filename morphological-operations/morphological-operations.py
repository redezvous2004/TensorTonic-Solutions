def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    # Write code here
    n, m = len(image), len(image[0])
    output = [[0] * m for _ in range(n)]
    pad_size = len(kernel) // 2
    for pixels in image:
        for _ in range(pad_size):
            pixels.insert(0, 0)
        pixels.extend([0] * pad_size)
    new_n, new_m = len(image), len(image[0])
    for _ in range(pad_size):
        image.insert(0, [0] * new_m)
        image.append([0] * new_m)
    for i in range(len(output)):
        for j in range(len(output[0])):
            for x in range(len(kernel)):
                flag = 0
                for y in range(len(kernel[0])):
                    if operation == "erode":
                        output[i][j] = 1
                        if kernel[x][y] == 1:
                            if image[i + x][j + y] == 0:
                                output[i][j] = 0
                                flag = 1
                                break
                    if operation == "dilate":
                        if kernel[x][y] == 1:
                            if image[i + x][j + y] == 1:
                                output[i][j] = 1
                                flag = 1
                                break
                if flag == 1:
                    break
    return output