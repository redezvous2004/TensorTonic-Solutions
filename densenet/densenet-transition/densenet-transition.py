import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    # YOUR CODE HERE
    x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight = map(lambda a: torch.tensor(a), [x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight])
    bn_gamma, bn_beta, bn_mean, bn_var = map(lambda a: a.reshape(1, -1, 1, 1), [bn_gamma, bn_beta, bn_mean, bn_var])
    norm_out = bn_gamma * ((x - bn_mean) / torch.sqrt(bn_var + eps)) + bn_beta
    relu_out = F.relu(norm_out)
    compression = F.avg_pool2d(F.conv2d(relu_out, conv_weight, padding=0), kernel_size=2)
    return compression
