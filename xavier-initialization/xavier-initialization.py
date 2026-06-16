import math
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    n, m = len(W), len(W[0])
    xavier_weights = [[0] * m for _ in range(n)]
    bound = math.sqrt(6 / (fan_in + fan_out))
    for i in range(n):
        for j in range(m):
            xavier_weights[i][j] = W[i][j] * 2 * bound - bound
    return xavier_weights