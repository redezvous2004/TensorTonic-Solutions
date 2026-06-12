def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    assignment = []
    for point in points:
        best_dist = float('inf')
        best_idx = 0
        for i in range(len(centroids)):
            dist = sum((point[j] - centroids[i][j]) ** 2 for j in range(len(centroids[i])))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        assignment.append(best_idx)
    return assignment
            