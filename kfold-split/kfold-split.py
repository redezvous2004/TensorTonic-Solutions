import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    idx = np.arange(N)
    if rng is None:
        np.random.shuffle(idx)
        shuffled_idx = idx
    else:
        shuffled_idx = rng.permutation(idx)

    folds = np.array_split(shuffled_idx, k)
    results = []
    for i in range(len(folds)):
        val = folds[i]
        train = np.concatenate(folds[:i] + folds[i + 1:])
        results.append((train, val))
    return results
        