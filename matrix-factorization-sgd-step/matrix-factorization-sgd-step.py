def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    # Write code here
    N = len(U)
    total = sum(U[i] * V[i] for i in range(N))
    error = r - total
    new_U = [U[i] + lr * (error * V[i] - reg * U[i]) for i in range(N)]
    new_V = [V[i] + lr * (error * U[i] - reg * V[i]) for i in range(N)]
    return new_U, new_V