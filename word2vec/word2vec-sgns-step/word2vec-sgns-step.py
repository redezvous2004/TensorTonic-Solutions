import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # YOUR CODE HERE
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()

    vec_center = W_in[center_id] #(d,)
    vec_pos = W_out[pos_id]
    vec_neg = W_out[neg_ids] # (k, d)

    norm_pos_score = torch.sigmoid(vec_pos @ vec_center)
    norm_neg_score = torch.sigmoid(vec_neg @ vec_center) # (k,)

    # Pos context
    grad_pos_score = (norm_pos_score - 1) * vec_center
    W_out_updated[pos_id] -= lr * grad_pos_score
    # Neg context
    grad_neg_score = norm_neg_score.unsqueeze(1) * vec_center.unsqueeze(0) # (k, d)
    W_out_updated.index_add_(0, neg_ids, -lr * grad_neg_score)
    # Center
    grad_center = (norm_pos_score - 1) * vec_pos + torch.sum(norm_neg_score.unsqueeze(1) * vec_neg, dim=0)
    W_in_updated[center_id] -= lr * grad_center
    return W_in_updated, W_out_updated
    


    
