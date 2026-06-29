def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    n = len(values)
    count = {}
    for value in values:
        count[value] = count.get(value, 0) + 1
    for k, v in count.items():
        count[k] = v / n
    result = []
    for value in values:
        result.append(count[value])
    return result