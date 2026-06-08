import math
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    dot_product, x1_cul, x2_cul = 0, 0, 0
    if len(x1) != len(x2):
        return None
    for i in range(len(x1)):
        dot_product += x1[i] * x2[i]
        x1_cul += x1[i] ** 2
        x2_cul += x2[i] ** 2
    x1_mag, x2_mag = math.sqrt(x1_cul), math.sqrt(x2_cul)
    cosin = dot_product / (x1_mag * x2_mag)
    if label == 1:
        return 1 - cosin
    else:
        return max(0, cosin - margin)