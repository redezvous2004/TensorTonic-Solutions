import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr, tpr = map(lambda x: np.asarray(x), [fpr, tpr])
    if len(fpr) != len(tpr) and len(fpr) < 2:
        return None
    return np.trapezoid(tpr, fpr)
    