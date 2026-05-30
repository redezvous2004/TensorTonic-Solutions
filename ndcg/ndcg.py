import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    len_sample = min(k, len(relevance_scores))
    relevance_scores_sample = relevance_scores[:len_sample]
    ideal_rel_scores = sorted(relevance_scores, reverse=True)[:len_sample]
    dcg_k, idcg_k = 0, 0
    for i in range(1, len(relevance_scores_sample) + 1):
        dcg_k += (math.pow(2, relevance_scores_sample[i - 1]) - 1) / math.log2(i + 1)
        idcg_k += (math.pow(2, ideal_rel_scores[i - 1]) - 1) / math.log2(i + 1)

    return dcg_k / idcg_k if idcg_k != 0 else 0.0
    