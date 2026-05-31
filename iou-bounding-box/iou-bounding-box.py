def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b
    x1_inter = max(x1a, x1b)
    y1_inter = max(y1a, y1b)
    x2_inter = min(x2a, x2b)
    y2_inter = min(y2a, y2b)
    
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    area_intersection =max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area_union = area_a + area_b - area_intersection

    iou = area_intersection / area_union if area_union != 0 else 0.0
    return iou