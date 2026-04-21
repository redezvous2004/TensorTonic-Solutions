import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    y_true, y_pred = map(lambda x: np.asarray(x), [y_true, y_pred])
    accuracy = float(np.mean(y_true==y_pred))

    if average=='binary':
        tp = np.sum((y_true == pos_label) & (y_true == y_pred))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    classes = np.unique(np.concatenate((y_true, y_pred)))
    tps = np.zeros(len(classes))
    fps = np.zeros(len(classes))
    fns = np.zeros(len(classes))
    supports = np.zeros(len(classes))

    for i, c in enumerate(classes):
        tps[i] = np.sum((y_true == c) & (y_pred == c))
        fps[i] = np.sum((y_true != c) & (y_pred == c))
        fns[i] = np.sum((y_true == c) & (y_pred != c))
        supports[i] = np.sum(y_true == c)
    if average == 'micro':
        total_tp = np.sum(tps)
        total_fp = np.sum(fps)
        total_fn = np.sum(fns)
        precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    elif average in ['macro', 'weighted']:
        precisions = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) != 0)
        recalls = np.divide(tps, tps + fns, out=np.zeros_like(tps), where=(tps + fns) != 0)
        f1s = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        if average == "macro":
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            f1 = float(np.mean(f1s))
            
        elif average == "weighted":
            total_support = np.sum(supports)
            if total_support > 0:
                precision = float(np.sum(precisions * supports) / total_support)
                recall = float(np.sum(recalls * supports) / total_support)
                f1 = float(np.sum(f1s * supports) / total_support)
            else:
                precision = recall = f1 = 0.0
        else:
            raise ValueError("Average mode must be one of: 'micro', 'macro', 'weighted', 'binary'.")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }