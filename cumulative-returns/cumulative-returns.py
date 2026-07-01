def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    cum_returns = []
    wealth = [1]
    for i in range(len(returns)):
        value = wealth[-1] * (1 + returns[i])
        wealth.append(value)
        cum_returns.append(value - 1)
    return cum_returns