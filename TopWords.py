import numpy as np

def topwords(W, n_components, n_top_words):
    num = n_components * n_top_words
    topwords = np.zeros(( num ))
    count = 0
    W = W.transpose()
    rows=W.shape[0]
    columns = W.shape[1]
    for i in range(0, rows):
        for j in W[i].argsort()[:-n_top_words - 1:-1]:
	        topwords[count] = int(j)
	        count = count + 1
    return topwords
