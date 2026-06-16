import math
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    n, m = len(W), len(W[0])
    he_weights = [[0] * m for _ in range(n)]
    bound = math.sqrt(6 / fan_in)
    for i in range(n):
        for j in range(m):
            he_weights[i][j] = W[i][j] * 2 * bound - bound
    return he_weights