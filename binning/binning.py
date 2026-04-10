def binning(values, num_bins):  
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    min_val = min(values)
    max_val = max(values)
    bin_idx = []
    bin_width = (max_val - min_val) / num_bins
    if bin_width == 0:
        return [0 for value in values]
    for value in values:
        bin = min(int((value - min_val) / bin_width), num_bins - 1)
        bin_idx.append(bin)
    return bin_idx