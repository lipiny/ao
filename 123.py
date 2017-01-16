# -*- coding: utf-8 -*-
#import io  
#import sys  
#import urllib.request  
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

from ReadTermList import readfile_termlist
from ReadTrInfoData import readfile_tr_infodata
from ReadTeInfoData import readfile_te_infodata
from sklearn import preprocessing
from nmf import nmf
from nmf_keepW import nmf_keepW
from TopWords import topwords
from BuildTopwordsMatrix import build_topwords_matrix
from nmfSetZero import nmf_set_zero
from knn import knn
from svm import svmclf
import time

#=====================parameter====================

	###
	#order = 'all'			parameter:none
	#order = 'continuous'	parameter:begin, rescale_num
	#order = 'backorder'	parameter:rescale_num
	#order = 'random'		parameter:begin, end, rescale_num
	###
#parameter for reading training data
tr_order = 'backorder'
tr_begin = 0
tr_end = 500
training_rescale_num = 50
#parameter for reading test data
te_order = 'continuous'
te_begin = 0
te_end = 100
test_rescale_num = 30
#---NMF
n_components=5
max_iter=30
#---knn
num_neighbors=5
#topwords
num_topwords = 20
#---file name
filename_termList = 'termList'
filename_training_info = 'train.info'
filename_test_info = 'test.info'
filename_training_data = 'train.data'
filename_test_data = 'test.data'
#=====================end para=====================


	#===============================
	#read term_list
	#return 2 1-dimension list: number of terms and the content of the term 
	#===============================

term_num, term_list = readfile_termlist(filename_termList)

	#===============================
	#read data&info file
	#===============================

print('-×-×-×-reading training data process begin-×-×-×-')
start_time = time.time()

### read training data
training_set, training_term_list, training_doc_num, training_title, training_categorie = readfile_tr_infodata(tr_order, tr_begin, tr_end, training_rescale_num, filename_training_info, filename_training_data)

end_time = time.time()
print('reading training data process finish, time: %f'%(end_time - start_time))
print('the new dictionary has %d words'%len(training_term_list))

print('-×-×-×-reading test data process begin-×-×-×-')
start_time = time.time()

### read test data
test_set, test_doc_num, test_title, test_categorie = readfile_te_infodata(te_order, te_begin, te_end, test_rescale_num, filename_test_info, filename_test_data, training_term_list)

end_time = time.time()
print('reading test data process finish time: %f'%(end_time - start_time))

training_set_normalize=preprocessing.normalize(training_set, axis=0)
test_set_normalize=preprocessing.normalize(test_set,axis=0)

	#===============================
	#nmf and knn
	#===============================

print('-×-×-×-NMF process begin-×-×-×-')
start_time = time.time()

### NMF
training_W, training_H = nmf(training_set_normalize, n_components, max_iter)

end_time = time.time()
print('NMF process end, time: %f'%(end_time - start_time))


test_H = nmf_keepW(test_set_normalize, training_W)

training_H_trans = training_H.transpose()
test_H_trans = test_H.transpose()

knn(num_neighbors, training_H_trans, training_categorie, test_H_trans, test_categorie)

svmclf(training_H_trans, training_categorie, test_H_trans, test_categorie)

	#===============================
	#new method
	#===============================

'''
top_words = topwords(training_W, n_components, num_topwords)

training_set_normalize_trans = training_set_normalize.tocsr()
test_set_normalize_trans = test_set_normalize.tocsr()
training_topwords_matrix = build_topwords_matrix(training_set_normalize_trans, top_words, training_doc_num)
test_topwords_matrix = build_topwords_matrix(test_set_normalize_trans, top_words, test_doc_num)

training_topwords_set_normalize=preprocessing.normalize(training_topwords_matrix, axis=0)
test_topwords_set_normalize=preprocessing.normalize(test_topwords_matrix, axis=0)

tw_training_W, tw_training_H = nmf(training_topwords_set_normalize, n_components)
tw_test_H = nmf_keepW(test_topwords_set_normalize, tw_training_W, n_components)

tw_training_H_trans = tw_training_H.transpose()
tw_test_H_trans = tw_test_H.transpose()

knn(num_neighbors, tw_training_H_trans, training_categorie, tw_test_H_trans, test_categorie)



tw_training_W, tw_training_H = nmf_set_zero(training_topwords_set_normalize, n_components)
tw_test_H = nmf_keepW(test_topwords_set_normalize, tw_training_W, n_components)

tw_training_H_trans = tw_training_H.transpose()
tw_test_H_trans = tw_test_H.transpose()

knn(num_neighbors, tw_training_H_trans, training_categorie, tw_test_H_trans, test_categorie)
'''
