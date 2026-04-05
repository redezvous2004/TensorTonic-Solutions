import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    final_batch = []
    for seq in seqs:
        seq = np.asarray(seq)
        diff = max_len - len(seq)
        if diff > 0:
            p_seq = np.pad(seq, (0, diff), mode='constant', constant_values=pad_value)
            final_batch.append(p_seq)
        else:
            final_batch.append(seq[:max_len])
    return np.asarray(final_batch)