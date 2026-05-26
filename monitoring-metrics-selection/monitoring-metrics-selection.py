import math
def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    results = []
    n = len(y_true)
    if system_type == "classification":
        tp, tn, fn = 0, 0, 0
        for tr, pred in zip(y_true, y_pred):
            if tr == pred and pred == 1:
                tp += 1
            elif tr == pred and pred == 0:
                tn += 1
            elif tr != pred and tr == 1 and pred == 0:
                fn += 1
        fp = n - tp - tn - fn
        accuracy = (tp + tn) / n if n != 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        results.extend([("f1", f1), ("accuracy", accuracy), ("recall", recall), ("precision", precision)])
    elif system_type == "regression":
        sub_arr = []
        for tr, pred in zip(y_true, y_pred):
            sub_arr.append(tr - pred)
        mae = (1 / n) * sum(abs(value) for value in sub_arr)
        rmse = math.sqrt((1 / n) * sum(value * value for value in sub_arr))
        results.extend([("mae", mae), ("rmse", rmse)])
    else:
        total_relevant = sum(val for val in y_true)
        arr = zip(y_pred, y_true)
        arr = sorted(arr, key=lambda a: -a[0])
        top_3 = sum(tr for pred, tr in arr[:3])
        precision_at_3 = top_3 / 3
        recall_at_3 = top_3 / total_relevant if total_relevant != 0 else 0.0
        results.extend([("precision_at_3", precision_at_3), ("recall_at_3", recall_at_3)])
    results = sorted(results, key=lambda x: x[0])
    return results