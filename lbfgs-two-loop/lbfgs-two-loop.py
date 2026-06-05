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
    
    alpha = [0.0] * m
    rho = [0.0] * m
    
    q = list(grad)
    
    for i in reversed(range(m)):
        s_i = s_list[i]
        y_i = y_list[i]
        
        ys_dot = _dot(y_i, s_i)
        if ys_dot == 0:
            rho[i] = 0.0
        else:
            rho[i] = 1.0 / ys_dot
            
        alpha[i] = rho[i] * _dot(s_i, q)

        q = [q_j - alpha[i] * y_ij for q_j, y_ij in zip(q, y_i)]
        
    s_last = s_list[-1]
    y_last = y_list[-1]
    yy_dot = _dot(y_last, y_last)
    
    gamma = _dot(s_last, y_last) / yy_dot if yy_dot != 0 else 1.0
    
    r = [gamma * q_j for q_j in q]
    
    for i in range(m):
        s_i = s_list[i]
        y_i = y_list[i]
        
        beta = rho[i] * _dot(y_i, r)
        
        coeff = alpha[i] - beta
        r = [r_j + coeff * s_ij for r_j, s_ij in zip(r, s_i)]
        
    direction = [-r_j for r_j in r]
    
    return direction