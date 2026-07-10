import numpy as np
def relu(x):
    return np.maximum(x, 0)
class BasicBlock:
    def __init__(self, W1, W2, W_proj = None):
        self.W_proj = W_proj
        self.W1 = W1
        self.W2 = W2
    def forward(self, x):
        out = relu(x @ self.W1)
        out = out @ self.W2
        if self.W_proj is not None:
            shortcut = x @ self.W_proj
        else:
            shortcut = x
        return relu(out + shortcut)
def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    # YOUR CODE HERE
    x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc = map(lambda a: np.asarray(a), [x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc])
    out = relu(x @ conv1)
    
    block1 = BasicBlock(W1_b1, W2_b1)
    out = block1.forward(out)

    block2 = BasicBlock(W1_b2, W2_b2, Ws_b2)
    out = block2.forward(out)

    logits = out @ fc
    return logits
