def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    if not values:
        return []
    results = list(values)
    left, right = -1e9, 1e9
    i = 1
    while i < len(values):
        if values[i] is None:
            if values[i - 1] is not None:
                left = i - 1
            i += 1
            continue
        else:
            if left != -1e9:
                for j in range(left + 1, i):
                    results[j] = values[left] + (j - left) / (i - left) * (values[i] - values[left])
        i += 1
    return results
            
            
        