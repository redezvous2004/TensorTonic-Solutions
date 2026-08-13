import numpy as np
import math

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE
    b, n, d = x.shape
    norm_x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6)
    
    Q = norm_x @ Wq
    K = norm_x @ Wk
    V = norm_x @ Wv

    head_dim = embed_dim // num_heads
    Q = Q.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_weight = Q @ K.transpose(0, 1, 3, 2) / head_dim ** 0.5
    norm_weight = attn_weight - np.max(attn_weight, axis=-1, keepdims=True)
    attn_weight = np.exp(norm_weight) / np.sum(np.exp(norm_weight), axis=-1, keepdims=True)
    attn_score = attn_weight @ V

    attn_score = attn_score.transpose(0, 2, 1, 3).reshape(b, n, -1)
    res = x + attn_score @ Wo
    norm_out = (res - np.mean(res, axis=-1, keepdims=True)) / (np.std(res, axis=-1, keepdims=True) + 1e-6)

    mlp_x = norm_out @ W1
    mlp_out = (0.5 * mlp_x * (1 + np.tanh(math.sqrt(2 / math.pi) * (mlp_x + 0.044715 * np.pow(mlp_x, 3))))) @ W2
    out = res + mlp_out
    return out
    