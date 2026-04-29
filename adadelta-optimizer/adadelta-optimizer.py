import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Write code here
    w, grad, E_grad_sq, E_update_sq = map(lambda x: np.asarray(x, dtype=float), [w, grad, E_grad_sq, E_update_sq])
    E_grad_sq = rho * E_grad_sq + (1 - rho) * (grad ** 2)
    delta_w = -(np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq + eps)) * grad
    E_update_sq = rho * E_update_sq + (1 - rho) * (delta_w ** 2)
    w = w + delta_w
    return w, E_grad_sq, E_update_sq