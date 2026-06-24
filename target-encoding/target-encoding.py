def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    cate_idx = {}
    for i in range(len(categories)):
        cate_idx[categories[i]] = cate_idx.get(categories[i], [])
        cate_idx[categories[i]].append(i)
    mean_target = {k: sum(targets[i] for i in v) / len(v) for k, v in cate_idx.items()}
    return [mean_target[cate] for cate in categories]
    
        
    
    
    