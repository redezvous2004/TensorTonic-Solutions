import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    # YOUR CODE
    x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight = map(lambda a: torch.tensor(a, dtype=torch.float64), [x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight])
    N, C, H, W = x.shape
    bn_gamma, bn_beta, bn_mean, bn_var = map(lambda a: a.reshape(1, -1, 1, 1), [bn_gamma, bn_beta, bn_mean, bn_var])
    norm_x = bn_gamma * ((x - bn_mean) / torch.sqrt(bn_var + eps)) + bn_beta
    return F.conv2d(F.relu(norm_x), conv_weight, stride=1, padding=1)
