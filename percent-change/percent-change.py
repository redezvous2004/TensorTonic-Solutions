def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    n = len(series)
    result = []
    for i in range(1, n):
        change = (series[i] - series[i - 1]) / series[i - 1] if series[i - 1] != 0 else 0.0
        result.append(change)
    return result