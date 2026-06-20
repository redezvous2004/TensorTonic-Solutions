def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here
    n = len(values)
    results = []
    for i in range(n - window_size + 1):
        sma = sum(values[i: i + window_size]) / window_size
        results.append(sma)
    return results