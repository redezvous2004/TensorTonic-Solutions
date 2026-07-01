def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    # Write code here
    level, trend = [series[0]], [series[1] - series[0]]
    for i in range(1, len(series)):
        level.append(alpha * series[i] + (1 - alpha) * (level[-1] + trend[-1]))
        trend.append(beta * (level[-1] - level[-2]) + (1 - beta) * trend[-1])
    return level