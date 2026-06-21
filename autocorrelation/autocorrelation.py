def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    n = len(series)
    mean = sum(seri for seri in series) / n
    gamma = sum((seri - mean) ** 2 for seri in series)
    lags = []
    for i in range(max_lag + 1):
        if i == 0:
            lags.append(1)
        else:
            rk = 0
            for j in range(n - i):
                rk += (series[j] - mean) * (series[j + i] - mean)
            rk = rk / gamma if gamma != 0 else 0.0
            lags.append(rk)
    return lags
                
    