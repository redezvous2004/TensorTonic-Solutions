import torch

def quantile_balancing(router_scores: torch.Tensor, current_bias: torch.Tensor, selected_count: int) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing routes, mixture weights, expert loads, and the next centered bias.
    """
    biased_scores = router_scores + current_bias
    sorted_scores = torch.argsort(biased_scores, dim=-1, descending=True, stable=True)
    selected_idx = sorted_scores[:, :selected_count]
    selected_scores = torch.gather(router_scores, dim=-1, index=selected_idx)
    mixture_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    loads = torch.bincount(selected_idx.reshape(-1), minlength=router_scores.shape[1])
    cutoffs = torch.gather(biased_scores, dim=-1, index=sorted_scores[:, selected_count: selected_count + 1])
    target_load = router_scores.shape[0] * selected_count // router_scores.shape[1]
    margins = router_scores - cutoffs
    ordered_margins = torch.sort(margins, dim=0, descending=True, stable=True).values
    uncentered_bias = -ordered_margins[target_load]
    next_bias = uncentered_bias - uncentered_bias.mean()
    return {"selected_experts": selected_idx, "mixture_weights": mixture_weights, "expert_loads": loads, "next_bias": next_bias}