import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    # YOUR CODE HERE
    n = random_tokens.shape[1]
    conditions = [
        (replace_probs  < 0.8) & mask_positions,
        (0.8 <= replace_probs) & (replace_probs < 0.9) & mask_positions,
        (replace_probs >= 0.9) & mask_positions
    ]
    choices = [
        mask_token_id,
        random_tokens,
        token_ids
    ]

    masked_ids = np.select(conditions, choices, default=token_ids)
    labels = np.where(mask_positions, token_ids, -100)
    return masked_ids, labels
    
class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        # YOUR CODE HERE
        logits = hidden_states @ self.W + self.b
        return logits
