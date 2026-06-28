def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    result = []
    for avg_rate, vote in items:
        weighted_rating = (vote / (vote + min_votes)) * avg_rate + (min_votes / (vote + min_votes)) * global_mean
        result.append(weighted_rating)
    return result