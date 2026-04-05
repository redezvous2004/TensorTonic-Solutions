import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))

    pos = np.arange(seq_len)[:, np.newaxis]

    i_even = np.arange(0, d_model, 2)
    i_odd = np.arange(1, d_model, 2)
    
    vals_even = np.exp(i_even * -(np.log(base)) / d_model)
    vals_odd = np.exp((i_odd - 1) * -(np.log(base)) / d_model)
    
    pe[:, 0::2] = np.sin(pos * vals_even)
    pe[:, 1::2] = np.cos(pos * vals_odd)


    return pe
    