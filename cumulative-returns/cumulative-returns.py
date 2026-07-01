def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    cum_returns = []
    wealth = []
    for i in range(len(returns)):
        if i == 0:
            wealth.append(1 + returns[i])
            cum_returns.append(returns[i])
        else:
            value = wealth[-1] * (1 + returns[i])
            wealth.append(value)
            cum_returns.append(value - 1)
    return cum_returns