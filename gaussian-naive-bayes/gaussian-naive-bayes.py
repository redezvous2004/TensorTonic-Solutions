import math
def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    classes = {}
    results = []
    n = len(X_train)
    for i in range(len(y_train)):
        classes[y_train[i]] = classes.get(y_train[i], [])
        classes[y_train[i]].append(X_train[i])
    priors = []
    for v in classes.values():
        priors.append(len(v) / n)
    means, vars = [], []
    for k, v in classes.items():
        n_element, features = len(v), len(v[0])
        mean_feature = [sum(v[i][j] for i in range(n_element)) / n_element for j in range(features)]
        var_feature = [sum((v[i][j] - mean_feature[j]) ** 2 for i in range(n_element)) / n_element for j in range(features)]
        means.append(mean_feature)
        vars.append([max(1e-9, var_value) for var_value in var_feature])
    for sample in X_test:
        best_idx, best_prob = -1, -1e9
        for idx, cls in enumerate(classes.keys()):
            log_prob = math.log(priors[idx])
            for i in range(len(sample)):
                log_density = -0.5 * math.log(2 * math.pi * vars[idx][i]) - ((sample[i] - means[idx][i]) ** 2 / (2 * vars[idx][i]))
                log_prob += log_density
            if best_prob < log_prob:
                best_prob = log_prob
                best_idx = idx
        results.append(list(classes.keys())[best_idx])
    return results
        