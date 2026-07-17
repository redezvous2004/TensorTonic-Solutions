import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE
    result = [stem_channels]
    for idx, n_layer in enumerate(block_layers):
        C = result[-1]
        C_block = C + n_layer * growth_rate
        result.append(C_block)
        if idx != len(block_layers) - 1:
            C_trans = math.floor(C_block * compression)
            result.append(C_trans)
    return torch.tensor(result, dtype=torch.int64)
