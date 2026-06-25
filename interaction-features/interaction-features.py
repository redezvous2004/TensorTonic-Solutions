def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    # Write code here
    n, m = len(X), len(X[0])
    results = []
    for i in range(n):
        features = [origin_feature for origin_feature in X[i]]
        for a in range(m):
            for b in range(a + 1, m):
                features.append(X[i][a] * X[i][b])
        results.append(features)
    return results