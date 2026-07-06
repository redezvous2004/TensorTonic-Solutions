import numpy as np
import math
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    seq_len_v = V.shape[1]

    d_head = d_model // num_heads

    Q_proj = Q @ W_q # (batch, seq_len, d_model)
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_head = Q_proj.reshape(batch, seq_len_q, num_heads, d_head).transpose(0, 2, 1, 3)
    K_head = K_proj.reshape(batch, seq_len_k, num_heads, d_head).transpose(0, 2, 1, 3)
    V_head = V_proj.reshape(batch, seq_len_v, num_heads, d_head).transpose(0, 2, 1, 3)

    attention_score = Q_head @ K_head.transpose(0, 1, 3, 2)
    attention_score = softmax(attention_score / math.sqrt(d_head))
    attention_weight = attention_score @ V_head

    attention_weight_concat = attention_weight.transpose(0, 2, 1, 3).reshape(batch, seq_len_q, -1)

    return attention_weight_concat @ W_o