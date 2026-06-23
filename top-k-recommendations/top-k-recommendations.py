def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    tup_scores = [(score, i) for i, score in enumerate(scores)]
    tup_scores.sort(key=lambda x: -x[0])
    results = []
    for score, idx in tup_scores:
        if idx in rated_indices:
            continue
        else:
            if len(results) == k:
                break
            results.append(idx)
    return results