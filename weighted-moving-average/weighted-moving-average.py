def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    n, k = len(values), len(weights)
    total_w = sum(weights)
    wma = []
    for i in range(n - k + 1):
        weighted_sum = 0
        for j in range(k):
            weighted_sum += weights[j] * values[i + j]
        wma.append(weighted_sum / total_w if total_w != 0 else 0.0)
    return wma