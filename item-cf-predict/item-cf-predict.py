def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    # Write code here
    sum_prods = sum(user_ratings[i] * item_similarities[i] for i in range(len(user_ratings)) if i != target and item_similarities[i] > 0 and user_ratings[i] != 0)
    sum_sim = sum(item_similarities[i] for i in range(len(item_similarities)) if i != target and item_similarities[i] > 0 and user_ratings[i] != 0)
    pred = sum_prods / sum_sim if sum_sim != 0 else 0.0
    return pred