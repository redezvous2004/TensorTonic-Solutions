import math
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    K = len(predictions)
    smoothed_target = []
    for i in range(K):
        if i == target:
            smoothed_label = (1.0 - epsilon) + (epsilon / K)
        else:
            smoothed_label = epsilon / K
        smoothed_target.append(smoothed_label)
    loss = 0
    for i in range(K):
        loss += -(smoothed_target[i] * math.log(predictions[i]))
    return loss