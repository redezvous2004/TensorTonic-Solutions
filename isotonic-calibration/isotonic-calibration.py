def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration using pure Python.
    """
    results = []
    cal_data = zip(cal_labels, cal_probs)
    sorted_data = sorted(cal_data, key=lambda x: x[1])
    cal_labels = [x[0] for x in sorted_data]
    cal_probs = [x[1] for x in sorted_data]
    
    n = len(cal_labels)
    if n > 0:
        pools = [[float(cal_labels[i]), 1] for i in range(n)]
        i = 0
        while i < len(pools) - 1:
            if (pools[i][0] / pools[i][1]) > (pools[i + 1][0] / pools[i + 1][1]):
                pools[i][0] += pools[i + 1][0]
                pools[i][1] += pools[i + 1][1]
                pools.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        idx = 0
        for pool in pools:
            avg_val = pool[0] / pool[1]
            for _ in range(pool[1]):
                cal_labels[idx] = avg_val
                idx += 1

    for prob in new_probs:
        if prob <= cal_probs[0]:
            results.append(cal_labels[0])
            continue
        if prob >= cal_probs[-1]:
            results.append(cal_labels[-1])
            continue
            
        for i in range(1, len(cal_probs)):
            if prob <= cal_probs[i]:
                p_i = cal_probs[i - 1]
                p_next = cal_probs[i]
                c_i = cal_labels[i - 1]
                c_next = cal_labels[i]
                
                if p_next - p_i == 0:
                    q_transform = c_i
                else:
                    q_transform = c_i + ((prob - p_i) / (p_next - p_i)) * (c_next - c_i)
                    
                results.append(q_transform)
                break 
                
    return results