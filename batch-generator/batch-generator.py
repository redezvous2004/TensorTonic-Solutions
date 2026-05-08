import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X, y = map(lambda a: np.asarray(a), [X, y])
    if rng is None:
        rng = np.random
    indices = np.arange(len(X))
    rng.shuffle(indices)
    for i in range(0, len(X), batch_size):
        X_batch_indices = indices[i: i + batch_size]
        y_batch_indices = indices[i: i + batch_size]
        if i + batch_size <= len(X):
            yield (X[X_batch_indices], y[y_batch_indices])
        else:
            if drop_last:
                continue
            else:
                yield (X[X_batch_indices], y[y_batch_indices])
