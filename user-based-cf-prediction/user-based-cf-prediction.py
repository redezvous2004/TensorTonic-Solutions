def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    # Write code here
    pos_idx = [i for i, sim in enumerate(similarities) if sim > 0]
    numer = sum(similarities[i] * ratings[i] for i in pos_idx)
    decor = sum(similarities[i] for i in pos_idx)
    weighted_avg = numer / decor if decor != 0 else 0.0
    return weighted_avg