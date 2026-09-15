import torch

def situ_glu(input_tensor: torch.Tensor, gate_projection: torch.Tensor, up_projection: torch.Tensor, gate_cap: float = 4.0, up_cap: float = 25.0) -> torch.Tensor:
    """
    Returns the bounded element-wise gated activation tensor.
    """
    g = input_tensor @ gate_projection
    u = input_tensor @ up_projection
    out = (gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)) * (up_cap * torch.tanh(u / up_cap))
    return out