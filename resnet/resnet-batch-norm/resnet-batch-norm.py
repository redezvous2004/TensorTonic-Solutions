import numpy as np
class BatchNorm:
    def __init__(self, num_features, gamma, beta, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.gamma = gamma
        self.beta = beta
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean 
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_norm + self.beta

def relu(x):
    return np.maximum(0, x)
def post_activation_block(x, W1, W2, bn1, bn2, training = True):
    out = x @ W1
    out = bn1.forward(out, training)
    out = relu(out)

    out = out @ W2
    out = bn2.forward(out, training)
    out = out + x
    return relu(out)
def pre_activation_block(x, W1, W2, bn1, bn2, training = True):
    out = bn1.forward(x, training)
    out = relu(out)
    out = out @ W1

    out = bn2.forward(out, training)
    out = relu(out)
    out = out @ W2
    return out + x
def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    x, W1, W2, gamma1, beta1, gamma2, beta2 = map(lambda a: np.asarray(a), [x, W1, W2, gamma1, beta1, gamma2, beta2])
    batch, num_features = x.shape
    bn1 = BatchNorm(num_features, gamma1, beta1)
    bn2 = BatchNorm(num_features, gamma2, beta2)
    if mode == "pre":
        result = pre_activation_block(x, W1, W2, bn1, bn2)
    else:
        result = post_activation_block(x, W1, W2, bn1, bn2)
    result = np.round(result, 4)
    return {
        'output': result,
        'mode': mode
    }