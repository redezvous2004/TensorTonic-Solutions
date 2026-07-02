def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    unique_recs = set([element for row in recommendations for element in row])
    return len(unique_recs) / n_items