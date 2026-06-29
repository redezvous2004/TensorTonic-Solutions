def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    order_mapping = {order: i for i, order in enumerate(ordering)}
    result = []
    for value in values:
        result.append(order_mapping[value])
    return result