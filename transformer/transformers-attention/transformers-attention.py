import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    batch, seq_len, d_k = Q.shape
    attention_score = Q @ K.transpose(-2, -1)
    attention_score = F.softmax(attention_score / math.sqrt(d_k), dim=-1)
    attention_weight =  attention_score @ V
    return attention_weight