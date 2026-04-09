import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X, y = map(lambda x: np.asarray(x), [X, y])
    _, counts = np.unique(y, return_counts=True)

    parent_gini = 1 - np.sum((counts / y.shape[0]) ** 2)
    best_feature, best_threshold, best_gain = 0, 0, 0
    for j in range(X.shape[1]):
        features_values = X[:, j]
        features_values_sorted = np.unique(features_values)
        threshold_values = features_values_sorted[:-1] + np.diff(features_values_sorted) / 2
        for threshold in threshold_values:
            mask = features_values <= threshold
            
            left, y_left = features_values[mask], y[mask]
            right, y_right = features_values[~mask], y[~mask]
            
            _, left_counts = np.unique(y_left, return_counts=True)
            _, right_counts = np.unique(y_right, return_counts=True)
            
            left_gini = 1 - np.sum((left_counts / y_left.shape[0]) ** 2)
            right_gini = 1 - np.sum((right_counts / y_right.shape[0]) ** 2)

            gain = parent_gini - (y_left.shape[0] / len(y)) * left_gini - (y_right.shape[0] / len(y)) * right_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = threshold
                
    return [best_feature, best_threshold]
            
            
        