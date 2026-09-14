import torch

def full_attention_residual(embedding: torch.Tensor, previous_outputs: torch.Tensor, pseudo_query: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing retrieved representations and depth-attention weights.
    """
    # embedding: (batch, seq, d_model), previous_outputs: (layers, batch, seq, d_model), pseudo_query: (d_model,)
    sources = torch.cat((embedding.unsqueeze(0), previous_outputs), dim=0)
    normed_sources = sources / torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)

    logits = (normed_sources * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)

    return {
        "retrieved_representation": retrieved,
        "attention_weights": weights
    }