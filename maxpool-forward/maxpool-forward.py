def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    h, w = len(X), len(X[0])
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    outputs = [[0] * w_out for _ in range(h_out)]
    for i in range(h_out):
        for j in range(w_out):
            best_value = -1e9 - 1
            for a in range(pool_size):
                for b in range(pool_size):
                    best_value = max(best_value, X[i * stride + a][j * stride + b])
            outputs[i][j] = best_value
    return outputs