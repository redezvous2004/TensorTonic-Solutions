import math
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    result = []
    for value in values:
        result.append(math.log(value + 1))
    return result