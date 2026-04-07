def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    k = min(k, len(recommended))
    top_k = recommended[:k]
    relevant_item = [item for item in top_k if item in relevant]

    precision_at_k = len(relevant_item) / k
    recall_at_k = len(relevant_item) / len(relevant)

    return [precision_at_k, recall_at_k]
    