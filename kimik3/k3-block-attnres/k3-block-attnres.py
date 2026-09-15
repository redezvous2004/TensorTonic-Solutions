import torch

def _read_depth_sources(sources, pseudo_query, eps):
    normalized = sources / torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)
    logits = (normalized * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights

def block_attention_residual(embedding: torch.Tensor, previous_outputs: torch.Tensor, pseudo_query: torch.Tensor, block_size: int, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing the retrieved values, depth weights, and block sources.
    """
    # previous_outputs: (num_layers, seq, d_model)
    num_layers = previous_outputs.shape[0]
    num_incompleted_layers = num_layers % block_size
    num_completed_layers = num_layers - num_incompleted_layers
    completed_blocks = [
        previous_outputs[i: i + block_size, ...].sum(dim=0)
        for i in range(0, num_completed_layers, block_size)
    ]

    sources = [embedding, *completed_blocks]
    if num_completed_layers < num_layers:
        sources.append(previous_outputs[num_completed_layers:].sum(dim=0))
    final_sources = torch.stack(sources)
    retrieved, weights = _read_depth_sources(final_sources, pseudo_query, eps)
    return {
        "retrieved_representation": retrieved,
        "attention_weights": weights,
        "block_sources": final_sources
    }
    