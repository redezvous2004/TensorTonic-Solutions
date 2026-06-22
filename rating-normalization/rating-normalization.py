def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    mean_ratings = []
    
    n, m = len(matrix), len(matrix[0])
    for i in range(n):
        total, rated = 0, 0
        for j in range(m):
            if matrix[i][j] != 0:
                total += matrix[i][j]
                rated += 1
        if total == 0:
            mean_ratings.append(0)
        else:
            mean_ratings.append(total / rated)
    outputs = matrix
    for i in range(n):
        if mean_ratings[i] == 0:
            continue
        else:
            for j in range(m):
                if matrix[i][j] != 0:
                    outputs[i][j] -= mean_ratings[i]
    return outputs