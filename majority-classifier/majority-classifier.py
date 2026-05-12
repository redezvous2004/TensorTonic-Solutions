import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train, X_test = map(lambda x: np.asarray(x), [y_train, X_test])
    labels, count = np.unique(y_train, return_counts=True)
    max_freq_idx = np.argmax(count)
    return np.full(X_test.shape, labels[max_freq_idx])