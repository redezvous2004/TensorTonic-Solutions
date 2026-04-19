import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X, labels = map(lambda a: np.asarray(a), [X, labels])
    clusters = np.unique(labels)
    n_samples = len(X)
    euclid_distances = np.linalg.norm(X[:, np.newaxis, :] - X, axis=2) # (6, 6)
    scores = np.zeros(n_samples)
    for cluster in clusters:
        in_cluster = (labels == cluster)
        N = np.sum(in_cluster)
        if N == 1:
            scores[in_cluster] = 0.0
            continue
        a = np.sum(euclid_distances[in_cluster][:, in_cluster], axis=-1) / (N - 1)
        b = np.full(N, np.inf)
        for other_cluster in clusters:
            if other_cluster == cluster:
                continue
            other_in_cluster = (labels == other_cluster)
            mean_dist = np.mean(euclid_distances[in_cluster][:, other_in_cluster], axis=-1)
            b = np.minimum(b, mean_dist)
        scores[in_cluster] = (b - a)/ np.maximum(a, b)
    return np.mean(scores)
    