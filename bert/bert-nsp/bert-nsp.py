import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    """
    Returns: list of (sentence_A, sentence_B, is_next_label) tuples
    """
    # YOUR CODE HERE
    result = []
    for pair in pair_specs:
        doc_a, doc_b, sent_a, sent_b = pair["doc_a"], pair["doc_b"], pair["sent_a"], pair["sent_b"]
        result.append([documents[doc_a][sent_a], documents[doc_b][sent_b], 1 if (doc_a == doc_b) & (sent_a + 1 == sent_b) else 0])
    return result

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext logits: cls_hidden @ W + b
        """
        # YOUR CODE HERE
        logits = cls_hidden @ self.W + self.b
        return logits

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
