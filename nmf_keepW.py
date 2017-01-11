import numpy as np
from scipy import linalg

def nmf_keepW(X, W):

	eps = 1e-5
	X = X.toarray()
	print('hello mdzz')
	# W = W
	# H is W\X

	H = linalg.lstsq(W, X)[0]
	H = np.maximum(H, eps)
	return H
