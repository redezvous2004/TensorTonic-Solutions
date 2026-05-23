def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    word_count = {}
    for sentence in sentences:
        for token in sentence:
            word_count[token] = word_count.get(token, 0) + 1
    sorted_items = sorted(word_count.items(), key=lambda item: (-item[1], item[0]))
    sorted_dict = dict(sorted_items)
    return sorted_dict