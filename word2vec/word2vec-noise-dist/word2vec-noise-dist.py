import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    # YOUR CODE HERE
    total_count = torch.sum(counts ** alpha)
    result = counts ** alpha / total_count
    return torch.Tensor(result)
