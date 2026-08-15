import numpy as np
import math

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        # Initialize weights here
        self.W_patch = W_patch
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.encoder_weights = encoder_weights
        self.W_head = W_head
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        # YOUR CODE HERE
        b, _, _, c = x.shape
        n = self.image_size ** 2 // self.patch_size ** 2
        patch_dim = self.patch_size * self.patch_size * c
        self.cls_token = np.tile(self.cls_token, (b, 1, 1))
        
        patched_x = x.reshape(b, self.image_size // self.patch_size, self.patch_size, self.image_size // self.patch_size, self.patch_size, c)
        patched_x = patched_x.transpose(0, 1, 3, 2, 4, 5)
        patched_x = patched_x.reshape(b, n, patch_dim)
        z = patched_x @ self.W_patch
        z = np.concatenate((self.cls_token, z), axis=1)
        z1 = z + self.pos_embed
        seq_len = n + 1
        
        for i in range(self.depth):
            weights = self.encoder_weights[i]
            # LayerNorm and MSA
            norm_z1 = (z1 - np.mean(z1, axis=-1, keepdims=True)) / (np.std(z1, axis=-1, keepdims=True) + 1e-6)
            Q = norm_z1 @ weights["Wq"]
            K = norm_z1 @ weights["Wk"]
            V = norm_z1 @ weights["Wv"]
    
            head_dim = self.embed_dim // self.num_heads
            Q = Q.reshape(b, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
            K = K.reshape(b, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
            V = V.reshape(b, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
    
            attn_weights = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
            norm_weights = attn_weights - np.max(attn_weights, axis=-1, keepdims=True)
            normed_weights = np.exp(norm_weights) / np.sum(np.exp(norm_weights), axis=-1, keepdims=True)
            attn_scores = normed_weights @ V
            attn_scores = attn_scores.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)
            res = z1 + attn_scores @ weights["Wo"]
    
            # LayerNorm and MLP
            norm_out = (res - np.mean(res, axis=-1, keepdims=True)) / (np.std(res, axis=-1, keepdims=True) + 1e-6)
            out1 = norm_out @ weights["W1"]
            gelu_out = 0.5 * out1 * (1 + np.tanh(math.sqrt(2 / math.pi) * (out1 + 0.044715 * np.pow(out1, 3))))
            out2 = gelu_out @ weights["W2"]
            z1 = res + out2

        norm_encoder_out = (z1 - np.mean(z1, axis=-1, keepdims=True)) / (np.std(z1, axis=-1, keepdims=True) + 1e-6)
        logits = norm_encoder_out[:, 0, :] @ self.W_head
        return logits
        
        