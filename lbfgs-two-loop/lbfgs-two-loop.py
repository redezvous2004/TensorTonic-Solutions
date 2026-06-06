def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    m = len(s_list)
    if m == 0:
        return [-g for g in grad]
    alpha, rho = [0.0] * m, [0.0] * m
    for i in reversed(range(m)):
        s_i = s_list[i]
        y_i = y_list[i]
        ys_dot = _dot(y_i, s_i)
        rho[i] = 1 / ys_dot if ys_dot != 0 else 0.0
        alpha[i] = rho[i] * _dot(s_i, grad)
        new_grad = [g - alpha[i] * y_ij for g, y_ij in zip(grad, y_i)]

    yy_dot = _dot(y_list[-1], y_list[-1])
    gamma = _dot(s_list[-1], y_list[-1]) / yy_dot if yy_dot != 0 else 1.0
    r = [gamma * g for g in new_grad]
    for i in range(m):
        s_i = s_list[i]
        y_i = y_list[i]
        beta = rho[i] * _dot(y_i, r)
        r = [r_j + s_ij * (alpha[i] - beta) for r_j, s_ij in zip(r, s_i)]
    return [-new_r for new_r in r]
    
    