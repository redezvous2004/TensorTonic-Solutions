import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    results = []
    for roi in rois:
        output = []
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1
        for i in range(output_size):
            row = []
            for j in range(output_size):
                h_start = y1 + math.floor(i * (roi_h / output_size))
                h_end = y1 + math.floor((i + 1) * (roi_h / output_size))
                if h_end == h_start:
                    h_end = h_start + 1
                w_start = x1 + math.floor(j * (roi_w / output_size))
                w_end = x1 + math.floor((j + 1) * (roi_w / output_size))
                if w_end == w_start:
                    w_end = w_start + 1
                max_val = -1e9 - 1
                for x in range(h_start, h_end):
                    for y in range(w_start, w_end):
                        max_val = max(max_val, feature_map[x][y])
                row.append(max_val)
            output.append(row)
        results.append(output)
    return results