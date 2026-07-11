import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    total = torch.sum(counts)
    words_freq = counts / total
    keep_probs = torch.clamp(torch.sqrt(t / words_freq), max=1.0)
    return keep_probs
