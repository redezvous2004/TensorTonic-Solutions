import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    values, transitions, rewards = map(lambda x: np.asarray(x), [values, transitions, rewards])
    sum_transitions = np.sum(transitions, axis=-1)
    updated_values = np.max(rewards + gamma * np.dot(transitions, values), axis=-1)
    return updated_values.tolist()