import math
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    N = len(predictions)
    focal_loss = []
    for i in range(N):
        if targets[i] == 1:
            p = predictions[i]
        else:
            p = 1 - predictions[i]
        focal_loss.append(-alpha * (1 - p) ** gamma * math.log(p))
    avg_fl = sum(fl for fl in focal_loss) / N
    return avg_fl
    