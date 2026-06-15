def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    n, d_in, d_out = len(X), len(X[0]), len(W[0])
    outputs = [[0] * d_out for _ in range(n)]
    for i in range(n):
        for j in range(d_out):
            total = 0
            for k in range(d_in):
                total += X[i][k] * W[k][j]
            outputs[i][j] = total + b[j]
    return outputs