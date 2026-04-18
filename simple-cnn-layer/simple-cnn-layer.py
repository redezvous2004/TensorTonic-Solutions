import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    padding, stride = 0, 1
    x, W, b = map(lambda a: np.asarray(a), [x, W, b])

    N, c_out = x.shape[0], W.shape[0]
    h_in, w_in, k_h, k_w = x.shape[-2], x.shape[-1], W.shape[-2], W.shape[-1]
    h_out = ((h_in + 2 * padding - k_h) // stride) + 1
    w_out = ((w_in + 2 * padding - k_w) // stride) + 1

    y = np.zeros((N, c_out, h_out, w_out))
    for n in range(N):
        for c in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    patch = x[n, :, i:i+k_h, j:j+k_w]
                    filter_w = W[c, :, :, :]
                    y[n, c, i, j] = np.sum(patch * filter_w) + b[c]
    return y
            
    