import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x)
    counter = Counter(x)
    max_freq = counter.most_common(1)[0][1]
    top_elements = [v for v, k in counter.items() if k == max_freq]
    return np.mean(x), np.median(x), min(top_elements)
    