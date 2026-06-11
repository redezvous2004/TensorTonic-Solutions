def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    td_errors = []
    V = 0
    for i in range(0, len(rewards)):
        td_errors.append(rewards[i] + gamma * values[i + 1] - values[i])
    A = 0
    advantages = []
    for td_error in reversed(td_errors):
        A = td_error + gamma * lam * A
        advantages.append(A)
    return list(reversed(advantages))
    