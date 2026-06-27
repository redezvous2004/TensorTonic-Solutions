def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    a, b = set(set_a), set(set_b)
    intersect = a & b
    union = a | b
    return len(intersect) / len(union) if len(union) != 0 else 0.0