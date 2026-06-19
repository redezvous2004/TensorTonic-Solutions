def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here
    h_out, w_out = len(X) // pool_size, len(X[0]) // pool_size
    outputs = [[-1e9] * w_out for _ in range(h_out)]
    for i in range(h_out):
        for j in range(w_out):
            total = 0
            for a in range(pool_size):
                for b in range(pool_size):
                    total += X[i * pool_size + a][j * pool_size + b]
            outputs[i][j] = total / pool_size ** 2
    return outputs