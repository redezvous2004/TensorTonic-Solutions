import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    q_values = np.asarray(q_values, dtype=float)
    if rng is None:
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(0, len(q_values))
        else:
            action_idx = np.argmax(q_values)
    else:
        if rng.random() < epsilon:
            action_idx = rng.integers(0, len(q_values))
        else:
            action_idx = np.argmax(q_values)
    return action_idx
    
