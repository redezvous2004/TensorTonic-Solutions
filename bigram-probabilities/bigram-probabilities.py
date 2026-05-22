def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    counts = {}
    for i in range(len(tokens) - 1):
        token1 = tokens[i]
        token2 = tokens[i + 1]
        pair = (token1, token2)
        counts[pair] = counts.get(pair, 0) + 1

    vocab = set(token for token in tokens)
    vocab_size = len(vocab)
    cw1 = {}
    for i in range(len(tokens) - 1):
        cw1[tokens[i]] = cw1.get(tokens[i], 0) + 1
    probs = {}
    for w1 in vocab:
        for w2 in vocab:
            pair = (w1, w2)
            count_pair = counts.get(pair, 0)
            count_w1 = cw1.get(w1, 0)
            probs[pair] = (count_pair + 1) / (count_w1 + vocab_size)
    return counts, probs
    