import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true, y_pred = map(lambda x: np.asarray(x), [y_true, y_pred])
    if num_classes is None:
        num_classes = max(y_true) + 1
    if num_classes == 1:
        return 1.0
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1
    if normalize == 'none':
        return confusion_matrix
    elif normalize == 'true':
        rows_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
        return confusion_matrix / rows_sum
    elif normalize == 'pred':
        cols_sum = np.sum(confusion_matrix, axis=0, keepdims=True)
        return confusion_matrix / cols_sum
    elif normalize == 'all':
        return confusion_matrix / len(y_true)