def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    # Write code here
    result = []
    max_lag = max(lags)
    for i in range(max_lag, len(series)):
        lag_values = [series[i - lag] for lag in lags]
        result.append(lag_values)
    return result