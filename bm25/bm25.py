import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    if len(docs) == 0:
        return np.array([])
    avg_len = sum(len(doc) for doc in docs) / len(docs)
    N = len(docs)
    df = Counter(word for doc in docs for word in set(doc))

    doc_scores = np.zeros((N,))
    for doc_idx, doc_words in enumerate(docs):
        term_counts = Counter(doc_words)
        doc_length = len(doc_words)
        score = []
        for word, count in term_counts.items():
            if word in query_tokens:
                idf = math.log(((N - df[word] + 0.5) / (df[word] + 0.5)) + 1)
                tf = (count * (k1 + 1)) / (count + k1 * (1 - b + b * (doc_length / avg_len)))
                score.append(idf * tf)
        doc_scores[doc_idx] = np.sum(score)
    return doc_scores