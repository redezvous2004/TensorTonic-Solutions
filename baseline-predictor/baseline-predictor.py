def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    # Write code here
    n, m = len(ratings_matrix), len(ratings_matrix[0])
    total, count = 0, 0
    for i in range(n):
        for j in range(m):
            if ratings_matrix[i][j] != 0:
                total += ratings_matrix[i][j]
                count += 1
    global_mean = total / count
    result = []
    for pair in target_pairs:
        user, item = pair
        user_rating, user_count, item_rated, item_count = 0, 0, 0, 0
        for j in range(m):
            if ratings_matrix[user][j] != 0:
                user_rating += ratings_matrix[user][j]
                user_count += 1
        for i in range(n):
            if ratings_matrix[i][item] != 0:
                item_rated += ratings_matrix[i][item]
                item_count += 1
        user_bias =  user_rating / user_count - global_mean
        item_bias = item_rated / item_count - global_mean
        result.append(global_mean + user_bias + item_bias)
    return result
        