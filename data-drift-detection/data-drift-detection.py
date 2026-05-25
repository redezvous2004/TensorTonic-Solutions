def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    ref_sum = sum(dist for dist in reference_counts)
    prod_sum = sum(dist for dist in production_counts)
    for i in range(len(reference_counts)):
        reference_counts[i] /= ref_sum
    for i in range(len(production_counts)):
        production_counts[i] /= prod_sum
    results = 0
    for ref, prod in zip(reference_counts, production_counts):
        results += abs(ref - prod)
    results = results / 2
    return {"score": results, "drift_detected": True if results > threshold else False}