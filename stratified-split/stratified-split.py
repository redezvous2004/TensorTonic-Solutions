import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X, y = map(lambda a: np.asarray(a), [X, y])
    if rng is None:
        rng_to_use = np.random
    else:
        rng_to_use = rng
        
    train_indices = []
    test_indices = []
    classes = np.unique(y)
    for c in classes:
        c_indices = np.where(y==c)[0]
        rng.shuffle(c_indices)
        n_test = int(np.round(len(c_indices) * test_size))
        test_indices.extend(c_indices[:n_test])
        train_indices.extend(c_indices[n_test:])
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
    
    