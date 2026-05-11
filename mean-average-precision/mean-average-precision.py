import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    # Write code here
    max_len = max(len(row) for row in y_true_list)
    N = len(y_true_list)
    y_true_pad = np.zeros((N, max_len), dtype=float)
    y_score_pad = np.full((N, max_len), -np.inf, dtype=float)
    for i, (t_row, s_row) in enumerate(zip(y_true_list, y_score_list)):
        y_true_pad[i, :len(t_row)] = t_row
        y_score_pad[i, :len(s_row)] = s_row

    y_true_pad, y_score_pad = map(lambda x: np.asarray(x), [y_true_pad, y_score_pad])
    sort_idx = np.argsort(-y_score_pad, axis=-1)
    # Sort scores and labels
    y_true_sorted = np.take_along_axis(y_true_pad, sort_idx, axis=-1)
    y_score_sorted = np.take_along_axis(y_score_pad, sort_idx, axis=-1)
    # Culmulative relevant items
    relevant_items_at_i = np.cumsum(y_true_sorted, axis=-1)
    idx = np.arange(y_true_sorted[:, :k].shape[-1])
    P = relevant_items_at_i[:, :k] / (idx[np.newaxis, :] + 1)
    numerator = np.sum(P * y_true_sorted[:, :k], axis=-1)
    decorator = relevant_items_at_i[:, -1]
    AP = np.where(decorator != 0, numerator / decorator, 0.0)
    mAP = np.mean(AP)
    return mAP, AP