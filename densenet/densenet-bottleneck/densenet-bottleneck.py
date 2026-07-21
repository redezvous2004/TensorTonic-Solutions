import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    # YOUR CODE HERE
    x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight, bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight = map(lambda a: torch.tensor(a), [x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight, bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight])
    bn1_gamma, bn1_beta, bn1_mean, bn1_var, bn2_gamma, bn2_beta, bn2_mean, bn2_var = map(lambda a: a.reshape(1, -1, 1, 1), [bn1_gamma, bn1_beta, bn1_mean, bn1_var, bn2_gamma, bn2_beta, bn2_mean, bn2_var])
    b, c, h, w = x.shape
    norm_out1 = bn1_gamma * ((x - bn1_mean) / torch.sqrt(bn1_var + eps)) + bn1_beta #(b, c, h, w)
    relu_out1 = F.relu(norm_out1)
    out1 = F.conv2d(relu_out1, conv1_weight, padding=0) # (b, 4xgr, h, w)

    norm_out2 = bn2_gamma * ((out1 - bn2_mean) / torch.sqrt(bn2_var + eps)) + bn2_beta
    relu_out2 = F.relu(norm_out2)
    out2 = F.conv2d(relu_out2, conv2_weight, padding=1) # (b, gr, h, w)

    return out2
    
