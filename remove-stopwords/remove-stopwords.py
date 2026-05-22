def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    output = []
    for token in tokens:
        if token not in stopwords:
            output.append(token)
    return output