import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    
    y_true, y_pred = map(lambda x: np.asarray(x, dtype=float), [y_true, y_pred])

    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        lower_bound = boundaries[i]
        upper_bound = boundaries[i + 1]

        if i == n_bins - 1:
            bin_mask = (y_pred >= lower_bound) & (y_pred <= upper_bound)
        else:
            bin_mask = (y_pred >= lower_bound) & (y_pred < upper_bound)
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_pred[bin_mask])
            weight = bin_size / len(y_true)

            ece += weight * np.abs(bin_acc - bin_conf)
    return ece