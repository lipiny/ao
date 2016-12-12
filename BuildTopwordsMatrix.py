import numpy as np
from scipy.sparse import csr_matrix

def build_topwords_matrix(ori_matrix, top_words, arc_num):

    len_indptr = len(top_words) + 1
    indptr = np.zeros((len_indptr))
    for i in range(0, len(top_words)):
        indptr[i+1] = indptr[i] + ( ori_matrix.indptr[int(top_words[i])+1] - ori_matrix.indptr[int(top_words[i])] )

    indices = np.zeros((int(indptr[len_indptr-1])))
    data = np.zeros((int(indptr[len_indptr-1])))
    count = 0
    for i in range(0, len(top_words)):
        for j in range(0, int(indptr[i+1]-indptr[i])):
            indices[count] = ori_matrix.indices[int(ori_matrix.indptr[int(top_words[i])])+j]
            data[count] = ori_matrix.data[int(ori_matrix.indptr[int(top_words[i])])+j]
            #print('in loop --%d-- the value is %d' %(i,top_words[i]))
            #print(data[count])
            count = count + 1

    topwords_matrix_CSR_matrix = csr_matrix((data, indices, indptr), shape=(len(top_words), arc_num))

    return topwords_matrix_CSR_matrix
