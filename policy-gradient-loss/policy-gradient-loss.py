def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    discounted_return = []
    T = len(rewards)
    for i in range(T - 1, -1, -1):
        if i == T - 1:
            discounted_return.append(rewards[i])
        else:
            discounted_return.append(rewards[i] + gamma * discounted_return[-1])
    discounted_return = list(reversed(discounted_return))
    mean_return = sum(value for value in discounted_return) / T
    advantages = [value - mean_return for value in discounted_return]
    loss = -sum(log_prob * advantage for log_prob, advantage in zip(log_probs, advantages)) / T
    return loss