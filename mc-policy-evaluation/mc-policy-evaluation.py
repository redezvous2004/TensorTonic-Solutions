import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    episodes = np.asarray(episodes)
    V = np.zeros(n_states)
    counts = np.zeros(n_states)
    for episode in episodes:
        g = 0.0
        visited = np.full(n_states, -1e5)
        for j in range(len(episode) - 1, -1, -1):
            state, reward = episode[j]
            g = reward + gamma * g
            visited[state] = g
        for s in range(n_states):
            if visited[s] != -1e5:
                V[s] += visited[s]
                counts[s] += 1
    V = np.divide(V, counts, out=np.zeros_like(V), where=counts != 0)
    return V
            
