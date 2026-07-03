def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    # Write code here
    means = []
    result = ratings_matrix.copy()
    users, items = len(ratings_matrix), len(ratings_matrix[0])
    
    if mode == "user":
        for i in range(users):
            total, quant = 0, 0
            for j in range(items):
                if ratings_matrix[i][j] != 0:
                    total += ratings_matrix[i][j]
                    quant += 1
            means.append(total / quant if quant != 0 else 0.0)
        for i in range(users):
            for j in range(items):
                if result[i][j] == 0:
                    result[i][j] = means[i]
    
    else:
        for i in range(items):
            total, quant = 0, 0
            for j in range(users):
                if ratings_matrix[j][i] != 0:
                    total += ratings_matrix[j][i]
                    quant += 1
            means.append(total / quant if quant != 0 else 0.0)
        for i in range(items):
            for j in range(users):
                if result[j][i] == 0:
                    result[j][i] = means[i]
    return result
        