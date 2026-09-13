import torch

def kda_recurrence(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, decay_logits: torch.Tensor, write_strength: torch.Tensor, output_gate_logits: torch.Tensor, output_projection: torch.Tensor, initial_state: torch.Tensor, g_min: float = -5.0, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing sequence outputs and the final recurrent state.
    """
    batch, seq, head, d_k = query.shape
    state = initial_state.clone()
    outputs = []
    for i in range(seq):
        q_t = query[:, i, ...]
        k_t = key[:, i, ...]
        v_t = value[:, i, ...]
        alpha_t = torch.exp(g_min * torch.sigmoid(decay_logits[:, i, ...]))
        beta_t = write_strength[:, i, ...]

        decay = alpha_t.unsqueeze(-1) * state
        erase = beta_t.unsqueeze(-1) * (k_t.unsqueeze(-1) @ (k_t.unsqueeze(-1).mT @ decay))
        write = beta_t.unsqueeze(-1) * (k_t.unsqueeze(-1) @ v_t.unsqueeze(-2))
        state = decay - erase + write # b, h, d_k, d_v

        read = (q_t.unsqueeze(-2) @ state).squeeze(-2) # b, h, d_v
        norm = read / torch.sqrt(read.square().mean(dim=-1, keepdims=True) + eps)
        gated = torch.sigmoid(output_gate_logits[:, i, ...]) * norm
        merged = gated.reshape(batch, -1) # b, h * d_v
        outputs.append(merged @ output_projection.T)
    final = torch.stack(outputs, dim=1)
    return {"outputs": final, "final_state": state}
        
        

        