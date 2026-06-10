def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    discounted_return = []
    R = 0
    T = len(rewards)
    for reward in reversed(rewards):
        R = reward + gamma * R
        discounted_return.append(R)
    discounted_return = list(reversed(discounted_return))
    mean_return = sum(discounted_return) / T
    advantages = [value - mean_return for value in discounted_return]
    loss = -sum(log_prob * advantage for log_prob, advantage in zip(log_probs, advantages)) / T
    return loss