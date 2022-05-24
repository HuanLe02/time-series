import numpy as np

def split_pre_post(seq, n_steps):
    y = seq[n_steps:]
    X = np.empty([y.size, n_steps])

    # construct X row-by-row
    for i in range(y.size):
        X[i] = seq[i:i+n_steps]
        
    return X, y
    