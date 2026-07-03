import math
def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
    if len(recommendations) == 0:
        return 0.0
    surprise = []
    for i in range(len(recommendations)):
        popularity = item_counts[i] / n_users
        surprise.append(-math.log2(popularity))
    return  sum(surprise) / len(surprise)