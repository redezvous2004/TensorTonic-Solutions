def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    sorted_items = sorted(models, key = lambda item: (item["accuracy"], -item["latency"], item["timestamp"]))
    return sorted_items[-1]["name"]