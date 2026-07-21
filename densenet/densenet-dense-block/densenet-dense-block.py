import torch
import torch.nn.functional as F

def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    # YOUR CODE HERE
    x = torch.tensor(x)
    result = [x]
    for i, layer in enumerate(layers):
        inp = torch.cat(result, dim=1)
        bn_var, bn_beta, bn_mean, bn_gamma, conv_weight = layer["bn_var"], layer["bn_beta"], layer["bn_mean"], layer["bn_gamma"], layer["conv_weight"]
        bn_var, bn_beta, bn_mean, bn_gamma, conv_weight = map(lambda a: torch.tensor(a), [bn_var, bn_beta, bn_mean, bn_gamma, conv_weight])
        bn_var, bn_beta, bn_mean, bn_gamma = map(lambda a: a.reshape(1, -1, 1, 1), [bn_var, bn_beta, bn_mean, bn_gamma])
        norm_out = bn_gamma * ((inp - bn_mean) / torch.sqrt(bn_var + eps)) + bn_beta
        relu_out = F.relu(norm_out)
        out = F.conv2d(relu_out, conv_weight, padding=1)
        result.append(out)
    return torch.cat(result, dim=1)
