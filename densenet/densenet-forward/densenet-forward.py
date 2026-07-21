import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-3x3Conv (padding 1, no bias) producing growth_rate channels.
    """
    # YOUR CODE HERE
    x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight = map(lambda a: torch.tensor(a), [x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight])
    bn_gamma, bn_beta, bn_mean, bn_var = map(lambda a: a.reshape(1, -1, 1, 1), [bn_gamma, bn_beta, bn_mean, bn_var])
    norm_out = bn_gamma * ((x - bn_mean) / torch.sqrt(bn_var + eps)) + bn_beta
    relu_out = F.relu(norm_out)
    out = F.conv2d(relu_out, conv_weight, padding=1)
    return out

def dense_block(x, layers, eps):
    """
    Returns torch.Tensor: concat of x and every composite-layer output (channels grow by growth_rate per layer).
    """
    # YOUR CODE HERE
    x = torch.tensor(x)
    result = [x]
    for layer in layers:
        inp = torch.cat(result, dim=1)
        bn_var, bn_beta, bn_mean, bn_gamma, conv_weight = layer["bn_var"], layer["bn_beta"], layer["bn_mean"], layer["bn_gamma"], layer["conv_weight"]
        composite_val = composite_layer(inp, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps)
        result.append(composite_val)
    return torch.cat(result, dim=1)
    
def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-1x1Conv then 2x2 average pool with stride 2 (channels compressed, H and W halved).
    """
    # YOUR CODE HERE
    x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight = map(lambda a: torch.tensor(a), [x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight])
    bn_gamma, bn_beta, bn_mean, bn_var = map(lambda a: a.reshape(1, -1, 1, 1), [bn_gamma, bn_beta, bn_mean, bn_var])
    norm_out = bn_gamma * ((x - bn_mean) / torch.sqrt(bn_var + eps)) + bn_beta
    relu_out = F.relu(norm_out)
    out = F.conv2d(relu_out, conv_weight, padding=0)
    return F.avg_pool2d(out, kernel_size=2, stride=2)

def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    # YOUR CODE HERE
    x = torch.tensor(x) # (b, c_in, h, w)
    
    # Stem layer
    stem_weight = weights["stem_conv"]
    stem_weight = torch.tensor(stem_weight)
    stem_out = F.conv2d(x, stem_weight, padding=1) # (b, stem_channel, h, w)
    
    # FC attribute
    fc_bias, fc_weight = weights["fc_bias"], weights["fc_weight"]
    fc_bias, fc_weight = map(lambda a: torch.tensor(a), [fc_bias, fc_weight])

    # Final norm attribute
    final_var, final_beta, final_mean, final_gamma = weights["final_bn_var"], weights["final_bn_beta"], weights["final_bn_mean"], weights["final_bn_gamma"]
    final_var, final_beta, final_mean, final_gamma = map(lambda a: torch.tensor(a), [final_var, final_beta, final_mean, final_gamma])
    final_var, final_beta, final_mean, final_gamma = map(lambda a: a.reshape(1, -1, 1, 1), [final_var, final_beta, final_mean, final_gamma])

    out = stem_out
    n = len(weights["blocks"])
    for i, block in enumerate(weights["blocks"]):
        if i < n - 1:
            out = dense_block(out, block, eps)
            transition_attr = weights["transitions"][i]
            trans_var, trans_beta, trans_mean, trans_gamma, trans_weight = transition_attr["bn_var"], transition_attr["bn_beta"], transition_attr["bn_mean"], transition_attr["bn_gamma"], transition_attr["conv_weight"]
            out = transition_layer(out, trans_gamma, trans_beta, trans_mean, trans_var, trans_weight, eps)
        else:
            out = dense_block(out, block, eps)
    norm_out = final_gamma * ((out - final_mean) / torch.sqrt(final_var + eps)) + final_beta
    relu_out = F.relu(norm_out)
    pooled_out = relu_out.mean(dim=(2, 3))
    final_result = pooled_out @ fc_weight.T + fc_bias
    return final_result
        