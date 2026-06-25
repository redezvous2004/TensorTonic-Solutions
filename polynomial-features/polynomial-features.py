import math
def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    results = [[math.pow(value, j) for j in range(degree + 1)] for value in values]
    return results
            