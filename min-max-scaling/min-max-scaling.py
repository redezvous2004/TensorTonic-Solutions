def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    min_cols, max_cols = [], []
    n, m = len(data), len(data[0])
    for j in range(m):
        min_cols.append(min(data[i][j] for i in range(n)))
        max_cols.append(max(data[i][j] for i in range(n)))
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if max_cols[j] != min_cols[j]:
                result[i][j] = (data[i][j] - min_cols[j]) / (max_cols[j] - min_cols[j]) 
    return result
            