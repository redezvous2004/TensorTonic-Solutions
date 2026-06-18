import math
def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    # Write code here
    n_cand, n_ref = len(candidate), len(reference)
    if n_cand == 0:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        candidate_n_grams = {}
        reference_n_grams = {}
        for i in range(n_cand - n + 1):
            n_grams = tuple(candidate[i: i + n])
            candidate_n_grams[n_grams] = candidate_n_grams.get(n_grams, 0) + 1
        for i in range(n_ref - n + 1):
            n_grams = tuple(reference[i: i + n])
            reference_n_grams[n_grams] = reference_n_grams.get(n_grams, 0) + 1
        count = 0
        for k, v in candidate_n_grams.items():
            if k in reference_n_grams:
                count += min(v, reference_n_grams.get(k))
        precisions.append(count / (n_cand - n + 1))
    if min(precisions) == 0:
        return 0.0
    BP = math.exp(min(0, 1 - n_ref / n_cand))
    BLEU = BP * math.exp(sum(math.log(prec) for prec in precisions) / max_n)
    return BLEU
                