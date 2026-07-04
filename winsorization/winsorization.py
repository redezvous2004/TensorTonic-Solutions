import math
def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    # Write code here
    n = len(values)
    lowerbound_idx = (n - 1) * lower_pct / 100
    upperbound_idx = (n - 1) * upper_pct / 100

    lowerbound_value = values[math.floor(lowerbound_idx)] + (lowerbound_idx - math.floor(lowerbound_idx)) * (values[math.ceil(lowerbound_idx)] - values[math.floor(lowerbound_idx)])
    upperbound_value = values[math.floor(upperbound_idx)] + (upperbound_idx - math.floor(upperbound_idx)) * (values[math.ceil(upperbound_idx)] - values[math.floor(upperbound_idx)])

    result = []
    for value in values:
        if value < lowerbound_value:
            result.append(lowerbound_value)
            continue
        elif value > upperbound_value:
            result.append(upperbound_value)
        else:
            result.append(value)
    return result