def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    # Write code here
    result = []
    cycle = int(len(series) / period)
    for i in range(period):
        avg = sum(series[i + period * j] for j in range(cycle)) / cycle
        result.append(avg)
    return result
        
            
        