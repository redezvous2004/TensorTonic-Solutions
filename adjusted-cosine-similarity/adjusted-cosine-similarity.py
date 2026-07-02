def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    # Write code here
    n, m = len(ratings_matrix), len(ratings_matrix[0])
    means = []
    for i in range(n):
        rated_items = [ratings_matrix[i][k] for k in range(m) if ratings_matrix[i][k] > 0]
        mean = sum(rated_items) / len(rated_items) if rated_items else 0.0
        means.append(mean)
    adjusted_ratings = []
    for i in range(n):
        adjusted_ratings.append((ratings_matrix[i][item_i] - means[i], ratings_matrix[i][item_j] - means[i]))
    numerator = 0
    deno_A, deno_B = 0, 0
    for i in range(n):
        if ratings_matrix[i][item_i] > 0 and ratings_matrix[i][item_j] > 0:
            adj_i = ratings_matrix[i][item_i] - means[i]
            adj_j = ratings_matrix[i][item_j] - means[i]
            numerator += adj_i * adj_j
            deno_A += adj_i ** 2
            deno_B += adj_j ** 2
    denominator = deno_A ** 0.5 * deno_B ** 0.5
    sim = numerator / denominator if denominator != 0 else 0.0
    return sim