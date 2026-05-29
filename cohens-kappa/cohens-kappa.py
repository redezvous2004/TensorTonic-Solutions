import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1, rater2 = map(lambda x: np.asarray(x), [rater1, rater2])
    unique_label = set(rater1)
    n = len(rater1)
    p_o = np.sum(rater1 == rater2) / n
    p_e = 0
    for k in unique_label:
        p_e += (np.sum(rater1 == k) / n) * (np.sum(rater2 == k) / n)

    k = (p_o - p_e) / (1 - p_e) if p_e != 1.0 else 1.0
    return k