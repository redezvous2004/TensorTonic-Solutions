def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    # Write code here
    sorted_vals = sorted(values)
    def median_values(arr, start, end):
        length = end - start + 1
        center = start + (length // 2)
        if length % 2 == 0:
            med = (arr[center - 1] + arr[center]) / 2
            return med, center - 1
        else:
            med = arr[center]
            return med, center
    n = len(values)
    if n < 2:
        return [0.0] * n
    
    median, pos_med = median_values(sorted_vals, 0, n - 1)
    if n % 2 == 0:
        q1, pos_q1 = median_values(sorted_vals, 0, pos_med)
        q3, pos_q3 = median_values(sorted_vals, pos_med + 1, n - 1)
    else:
        q1, pos_q1 = median_values(sorted_vals, 0, pos_med - 1)
        q3, pos_q3 = median_values(sorted_vals, pos_med + 1, n - 1)
    result = []
    for value in values:
        result.append((value - median) / (q3 - q1) if q3 != q1 else value - median)
    return result