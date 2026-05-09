import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    return {
        "min": np.full(D, np.inf),
        "max": np.full(D, -np.inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    X_batch = np.asarray(X_batch)
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)
    
    state['min'] = np.minimum(state['min'], batch_min)
    state['max'] = np.maximum(state['max'], batch_max)

    X_norm = (X_batch - state["min"]) / (state["max"] - state["min"] + eps)
    return X_norm