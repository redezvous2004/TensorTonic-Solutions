import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    
    if rng is None:
        rng = np.random.default_rng()

    mask = (rng.random(x.shape) < (1 - p)).astype(x.dtype)
    dropout_pattern = mask / ((1 - p) if p < 1.0 else 1.0)
    output = x * dropout_pattern
    
    return output, dropout_pattern