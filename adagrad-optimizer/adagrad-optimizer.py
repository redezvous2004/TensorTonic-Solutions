import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w, g, G = map(lambda x: np.asarray(x, dtype=float), [w, g, G])
    G = G + g ** 2
    w = w - ((lr / np.sqrt(G + eps) * g))
    return w, G