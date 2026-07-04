def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    # Write code here
    n = len(values)
    sorted_arr = sorted(values)
    ranking_dict = {}
    for i in range(n):
        ranking_dict[sorted_arr[i]] = ranking_dict.get(sorted_arr[i], [])
        ranking_dict[sorted_arr[i]].append(i + 1)
    for k, v in ranking_dict.items():
        ranking_dict[k] = sum(v) / len(v)
    result = [ranking_dict[value] for value in values]
    return result