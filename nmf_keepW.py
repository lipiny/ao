import numpy as np
from scipy import linalg

def nmf_keepW(X, W, n_components, error_limit=1e-6):

    eps = 1e-5
    X = X.toarray()

    # W = W
    # H is W\X

    H = linalg.lstsq(W, X)[0]
    H = np.maximum(H, eps)
    return H
