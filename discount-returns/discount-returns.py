def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # Write code here
    results = []
    for i in range(len(rewards) - 1, -1, -1):
        if i == len(rewards) - 1:
            results.append(rewards[i])
        else:
            results.append(rewards[i] + gamma * results[-1])
    return list(reversed(results))
    