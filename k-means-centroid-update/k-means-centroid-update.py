def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    results = []
    clusters = {i: [] for i in range(k)}
    n_element = len(points[0])
    for i in range(len(assignments)):
        clusters[assignments[i]].append(points[i])
    for cluster, cluster_points in clusters.items():
        n_cluster = len(cluster_points)
        if n_cluster == 0:
            results.append([0.0] * n_element)
            continue  
        centroid = []
        for i in range(n_element):
            val = 0
            for j in range(n_cluster):
                val += cluster_points[j][i]
            centroid.append(val / n_cluster)
        results.append(centroid)
    return results