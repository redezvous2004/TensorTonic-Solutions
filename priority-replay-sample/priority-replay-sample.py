def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Write code here
    results = []
    powered_priors = [prior ** alpha for prior in priorities]
    probs = [prior / sum(powered_priors) for prior in powered_priors]
    results.append(probs)
    weights = [(len(priorities) * prob) ** -beta for prob in probs]
    normalized_weights = [weight / max(weights) for weight in weights]
    results.append(normalized_weights)
    return results
    