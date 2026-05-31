def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    # Write code here
    results = []
    mapping = [(idx, box, score) for idx, (box, score) in enumerate(zip(boxes, scores))]
    mapping.sort(key=lambda x: x[2], reverse=True)
    while mapping:
        idx, box1, score = mapping.pop(0)
        x1a, y1a, x2a, y2a = box1
        box1_area = (x2a - x1a) * (y2a - y1a)
        results.append(idx)
        for map in mapping[:]:
            _, box2, __ = map
            x1b, y1b, x2b, y2b = box2
            box2_area = (x2b - x1b) * (y2b - y1b)
            x1_inter = max(x1a, x1b)
            y1_inter = max(y1a, y1b)
            x2_inter = min(x2a, x2b)
            y2_inter = min(y2a, y2b)
            intersec_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
            union_area = box1_area + box2_area - intersec_area
            iou = intersec_area / union_area
            if iou >= iou_threshold:
                mapping.remove(map)
    return results
            
        
        
    