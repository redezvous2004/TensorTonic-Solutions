import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    elu_x = []
    for value in x:
        if value <= 0:
            elu_x.append(alpha * (math.exp(value) - 1))
        else:
            elu_x.append(value)
    return elu_x