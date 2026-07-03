def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    n, m = len(recommendations), len(recommendations[0])
    hit_rate = 0
    for i in range(n):
        for j in range(k):
            if recommendations[i][j] in ground_truth[i]:
                hit_rate += 1
                break
    return hit_rate / n