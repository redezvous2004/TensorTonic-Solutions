import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    X_train, y_train, X_test = map(lambda x: np.asarray(x), [X_train, y_train, X_test])
    vals = np.unique(y_train)
    n_test, n_classes = X_test.shape[0], len(vals)
    log_likelihoods = np.zeros((n_test, n_classes))
    for idx, c in enumerate(vals):
        X_c = X_train[y_train == c]

        prior = X_c.shape[0] / X_train.shape[0]
        log_prior = np.log(prior)

        p_ic = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)
        log_prob_features = X_test @ np.log(p_ic) + (1 - X_test) @ np.log(1 - p_ic)

        log_likelihoods[:, idx] = log_prior + log_prob_features
    return log_likelihoods
    
    