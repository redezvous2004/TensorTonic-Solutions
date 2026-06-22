def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    if not values:
        return []
    results = list(values)
    left = 0
    for i in range(1, len(values)):
        if results[i] is not None:
            if i - left > 1:
                y0, y1 = results[left], results[i]
                for j in range(left + 1, i):
                    results[j] = y0 + (j - left) * (y1 - y0) / (i - left)
            left = i          
    return results
            
            
        