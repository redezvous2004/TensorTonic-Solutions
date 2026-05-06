import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x, q = map(lambda a: np.asarray(a), [x, q])
    x = np.sort(x)
    idx = np.arange(len(x))
    pos = (q / 100) * (len(x) - 1)
    return np.interp(pos, idx, x)