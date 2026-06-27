import math
def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # Write code here
    stds = []
    n = len(values)
    for i in range(n - window_size + 1):
        mean = sum(values[i + j] for j in range(window_size)) / window_size
        std = math.sqrt(sum((values[i + j] - mean) ** 2 for j in range(window_size)) / window_size)
        stds.append(std)
    return stds