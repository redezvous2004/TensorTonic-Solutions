import math
import torch

def gated_mla(hidden_states: torch.Tensor, query_projection: torch.Tensor, latent_down_projection: torch.Tensor, key_up_projection: torch.Tensor, value_up_projection: torch.Tensor, output_gate_projection: torch.Tensor, output_projection: torch.Tensor, num_heads: int, causal: bool = True) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing gated attention output and the latent key-value cache.
    """
    batch, seq, d_model = hidden_states.shape

    query = hidden_states @ query_projection.T
    latent_vec = hidden_states @ latent_down_projection.T
    key = latent_vec @ key_up_projection.T
    value = latent_vec @ value_up_projection.T

    d_head = d_model // num_heads
    query = query.reshape(batch, seq, num_heads, d_head).transpose(1, 2)
    key = key.reshape(batch, seq, num_heads, d_head).transpose(1, 2)
    value = value.reshape(batch, seq, num_heads, d_head).transpose(1, 2)
    score = query @ key.transpose(-2, -1) / math.sqrt(d_head)
    if causal:
        mask = torch.triu(
            torch.ones_like(score, dtype=torch.bool, device=score.device),
            diagonal=1
        )
        score = score.masked_fill(mask, float("-inf"))
    weight = torch.softmax(score, dim=-1)
    ctx = weight @ value
    final = ctx.transpose(1, 2).reshape(batch, seq, -1)
    gated = torch.sigmoid(hidden_states @ output_gate_projection.T)
    output = (final * gated) @ output_projection.T

    return {"output": output, "latent_cache": latent_vec}