def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    # Write code here
    freqs = {}
    n, m = len(image), len(image[0])
    total_pixel = n * m
    output = [[0] * m for _ in range(n)]
    for pixels in image:
        for pixel in pixels:
            freqs[pixel] = freqs.get(pixel, 0) + 1
    sorted_freqs = dict(sorted(freqs.items(), key=lambda item: item[0]))
    key_arr = sorted_freqs.keys()
    val_arr = sorted_freqs.values()
    sum = 0
    cdf = []
    for val in val_arr:
        sum += val
        cdf.append(sum)
    cdf_dict = {k: v for k, v in zip(key_arr, cdf)}
    cdf_min = min(cdf_val for cdf_val in cdf_dict.values())
    for i in range(n):
        for j in range(m):
            output[i][j] = round(((cdf_dict[image[i][j]] - cdf_min) / (total_pixel - cdf_min)) * 255) if total_pixel != cdf_min else 0
    return output
        