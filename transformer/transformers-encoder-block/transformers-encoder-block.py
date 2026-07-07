import numpy as np
import math
def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    seq_len_v = V.shape[1]

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    d_head = d_model // num_heads
    Q_heads = Q_proj.reshape(batch, seq_len_q, num_heads, d_head).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch, seq_len_k, num_heads, d_head).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch, seq_len_v, num_heads, d_head).transpose(0, 2, 1, 3)

    attention_score = softmax(Q_heads @ K_heads.transpose(0, 1, 3, 2) / math.sqrt(d_head))
    attention_weigth = attention_score @ V_heads

    output = attention_weigth.transpose(0, 2, 1, 3).reshape(batch, seq_len_q, -1) @ W_o
    return output
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = x @ W1 + b1
    relu_out = np.maximum(0, hidden)
    output = relu_out @ W2 + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    residual_conn = x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_com = layer_norm(residual_conn, gamma1, beta1)
    ffn_out = feed_forward(x_com, W1, b1, W2, b2)
    output = layer_norm(x_com + ffn_out, gamma2, beta2)
    return output