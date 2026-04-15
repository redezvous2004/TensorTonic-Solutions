def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    N = len(y_true)
    tp, fp, fn = 0, 0, 0
    for i in range(N):
        if y_true[i] == y_pred[i]:
            tp += 1
        else:
            fp += 1
            fn += 1
    return (2 * tp) / (2 * tp + fp + fn)