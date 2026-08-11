import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    # Your implementation here
    generated_samples = np.asarray(generated_samples)
    diversity_score = np.mean(np.std(generated_samples, axis=0), axis=-1)
    return {
        "diversity_score": diversity_score,
        "is_collapsed": (diversity_score < threshold)
    }