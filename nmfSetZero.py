import numpy as np
from scipy import linalg
from numpy import dot
from sklearn import preprocessing

def nmf_set_zero(X, n_components, max_iter=50, error_limit=1e-6):

    eps = 1e-5
    print('Starting NMF decomposition with %d components and %d iterations.'%(n_components, max_iter))
    X = X.toarray()

    # initial. W is random [0,1] and H is W\X.
    rows, columns = X.shape
    W = np.random.rand(rows, n_components)
    W = np.maximum(W, eps)
    for i in range(0, rows - 1):
        for j in range(0, n_components):
            if (j != i and j != n_components - 1):
                W[i][j] = 0
    W=preprocessing.normalize(W, axis=0)

    H = linalg.lstsq(W, X)[0]
    H = np.maximum(H, eps)

    for i in range(1, max_iter + 1):
        # ===== updates =====
        # W=W.*(((W.*X)*H')./((W.*(W*H))*H'));
        top = dot(X, H.T)
        bottom = dot(dot(W, H), H.T) + eps
        W *= top / bottom

        for m in range(0, rows - 1):
            for n in range(0, n_components):
                if (n != m and n != n_components - 1):
                    W[m][n] = 0
        W=preprocessing.normalize(W, axis=0)

        # H=H.*((W'*(W.*X))./(W'*(W.*(W*H))));
        top = dot(W.T, X)
        bottom = dot(W.T, dot(W, H)) + eps
        H *= top / bottom
        H = np.maximum(H, eps)

        # ==== evaluation ====
        if i % 10 == 0 or i == 1 or i == max_iter:
            print('Iteration %d'%i)
            X_est = dot(W, H)
            curRes = linalg.norm((X - X_est), ord='fro')
            print('residual: %f'%np.round(curRes, 4))
            if curRes < error_limit:
                print('bbb out of limit')
                break
    return W, H
