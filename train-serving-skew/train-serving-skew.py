import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    # Write code here
    keys = train_dist.keys()
    results = {}
    for key in keys:
        if key not in serving_dist.keys():
            continue
        p_train = np.asarray(train_dist[key], dtype=float) + eps
        p_serving = np.asarray(serving_dist[key], dtype=float) + eps
        terms = (p_serving - p_train) * np.log(p_serving / p_train)
        psi = np.sum(terms)
        feature_score = {"psi": psi, "skewed": True if psi >= threshold else False}
        results.update({key: feature_score})
    return results
        
        