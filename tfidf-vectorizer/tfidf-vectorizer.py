import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    tokenized_docs = [doc.strip().split() for doc in documents]
    vocab = sorted(set([word.lower() for doc in tokenized_docs for word in doc]))
    word2idx = {word: i for i, word in enumerate(vocab)}

    N = len(documents)
    df = Counter(word.lower() for doc in tokenized_docs for word in set(doc))

    tfidf_matrix = np.zeros((N, len(vocab)))
    for doc_idx, doc_words in enumerate(tokenized_docs):
        term_counts = Counter(doc_words)
        total_terms = len(doc_words)
        for word, count in term_counts.items():
            tf = count / total_terms
            idf = math.log(N / df[word]) if df[word] != N else 0
            tfidf_matrix[doc_idx, word2idx[word]] = tf * idf
    return tfidf_matrix, vocab
 