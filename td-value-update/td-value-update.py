import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    V_copy = np.asarray(V, dtype=float).copy()
    delta = r + gamma * V_copy[s_next] - V_copy[s]
    V_copy[s] = V_copy[s] + alpha * delta
    return V_copy