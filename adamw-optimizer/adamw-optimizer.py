import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w, m, v, grad = map(lambda x: np.asarray(x, dtype=float), [w, m, v, grad])
    first_moment = beta1 * m + (1 - beta1) * grad
    second_moment = beta2 * v + (1 - beta2) * (grad ** 2)
    w = w - lr * (weight_decay * w) - lr * (first_moment / (np.sqrt(second_moment) + eps))
    return w, first_moment, second_moment