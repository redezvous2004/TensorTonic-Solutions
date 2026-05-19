import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    for episode in episodes:
        g = 0.0
        first_visit_g = np.full(n_states, -1e5)
        for j in range(len(episode) - 1, -1, -1):
            state, reward = episode[j]
            g = reward + gamma * g
            first_visit_g[state] = g
        for s in range(n_states):
            if first_visit_g[s] != -1e5:
                V[s] += first_visit_g[s]
                returns_count[s] += 1
    V = np.divide(V, returns_count, out=np.zeros_like(V), where=returns_count != 0)
    return V
            
