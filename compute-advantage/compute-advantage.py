import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    states, rewards, V = map(lambda x: np.asarray(x), [states, rewards, V])
    A = np.zeros(states.shape)
    g = 0.0
    for state in states[::-1]:
        g = rewards[state] + gamma * g
        A[state] = g - V[state]
    return A
    
