def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    # Write code here
    result = []
    for i in range(0, len(values) - window_size + 1):
        arr = values[i: i + window_size]
        arr.sort()
        center = window_size // 2
        if window_size % 2 == 0:
            result.append((arr[center - 1] + arr[center]) / 2)
        else:
            result.append(arr[center])
    return result