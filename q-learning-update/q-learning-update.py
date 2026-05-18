import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Write code here
    Q_copy = np.asarray(Q, dtype=float).copy()
    target = r + gamma * np.max(Q_copy[s_next, :]) - Q_copy[s, a]
    Q_copy[s, a] = Q_copy[s, a] + alpha * target
    return Q_copy