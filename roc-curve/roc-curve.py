import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # Write code here
    y_true, y_score = map(lambda x: np.asarray(x), [y_true, y_score])
    idx = np.lexsort((y_true, y_score))[::-1]
    y_true_sorted = y_true[idx]
    y_score_sorted = y_score[idx]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    diffs = np.diff(y_score_sorted)
    unique_threshold_idx = np.where(diffs != 0)[0]
    unique_threshold_idx = np.append(unique_threshold_idx, len(y_score_sorted) - 1)

    tps_at_thresholds = tps[unique_threshold_idx]
    fps_at_thresholds = fps[unique_threshold_idx]
    thresholds = y_score_sorted[unique_threshold_idx]

    tpr = tps_at_thresholds / tps[-1]
    fpr = fps_at_thresholds / fps[-1]
    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)
    thresholds = np.insert(thresholds, 0, np.inf)
    return fpr, tpr, thresholds
        
        
        