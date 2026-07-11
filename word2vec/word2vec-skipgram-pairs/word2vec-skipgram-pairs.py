import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    result = []
    n = len(token_ids)
    for i in range(len(token_ids)):
        for j in range(max(0, i - window), min(n - 1, i + window) + 1):
            if j != i:
                result.append([token_ids[i], token_ids[j]])
    if len(result) == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.Tensor(result).to(torch.int64)