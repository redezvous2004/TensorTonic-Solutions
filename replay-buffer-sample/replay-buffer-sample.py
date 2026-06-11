import numpy as np
def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    # Write code here
    buffer = np.asarray(buffer)
    rs = np.random.RandomState(seed)
    sampled_batch_idx = rs.choice(buffer.shape[0], size=batch_size, replace=False)
    sampled_batch = buffer[sampled_batch_idx]
    return sampled_batch