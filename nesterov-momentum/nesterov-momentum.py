import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w, v, grad = map(lambda x: np.asarray(x, dtype=float), [w, v, grad])
    v = momentum * v + lr * grad
    w = w - v
    return w, v