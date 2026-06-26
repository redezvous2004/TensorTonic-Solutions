def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code 
    result = series.copy()
    for _ in range(1, order + 1):
        order_result = []
        for j in range(len(result) - 1):
            order_result.append(result[j + 1] - result[j])
        result = order_result
    return result