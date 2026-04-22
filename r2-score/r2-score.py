import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true, y_pred = map(lambda x: np.asarray(x), [y_true, y_pred])
    y_mean = np.mean(y_true)
    unique_vals = np.unique(y_true)
    if len(unique_vals) == 1:   
        if np.array_equal(y_true, y_pred):
            return 1.0
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    return 1 - (ss_res / ss_tot)