import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    tokens, vocab = map(lambda x: np.asarray(x), [tokens, vocab])
    bow = np.zeros(len(vocab), dtype=int)
    mapping = {word: i for i, word in enumerate(vocab)}
    for token in tokens:
        if token in vocab:
            idx = mapping[token]
            bow[idx] += 1
    return bow