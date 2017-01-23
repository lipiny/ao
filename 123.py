# -*- coding: utf-8 -*-
#import io  
#import sys  
#import urllib.request  
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

from ReadTermInfo import readfile_terminfo
from ReadTermList import readfile_termlist
from ReadTrInfoData import readfile_tr_infodata
from ReadTeInfoData import readfile_te_infodata
from nmf_keepW import nmf_keepW
from facetW import facetW

from sklearn import preprocessing
from nmf import nmf

from WSetZero import W_set_zero
from knn import knn
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
training_rescale_num = 100
#parameter for reading test data
te_order = 'continuous'
te_begin = 0
te_end = 100
test_rescale_num = 100
#---NMF
n_components=10
max_iter=50
#---knn
num_neighbors=5
#topwords
num_topwords = 30
#---file name
filename_termInfo = 'termList.info'
filename_termList = 'termList'
filename_training_info = 'train.info'
filename_test_info = 'test.info'
filename_training_data = 'train.data'
filename_test_data = 'test.data'
#=====================end para=====================


	#===============================
	#read term_info
	#return the interval of each property 
	#read term_list
	#return 2 1-dimension list: number of terms and the content of the term 
	#===============================

term_interval = readfile_terminfo(filename_termInfo)
print(term_interval)
term_num, term_list = readfile_termlist(filename_termList)

	#===============================
	#read data&info file
	#===============================

print('-×-×-×-reading training data process begin-×-×-×-')
start_time = time.time()

### read training data
training_set, training_term_list, training_term_interval, training_doc_num, training_title, training_categorie = readfile_tr_infodata(tr_order, tr_begin, tr_end, training_rescale_num, term_interval, filename_training_info, filename_training_data)
#print(training_term_interval)

end_time = time.time()
print('reading training data process finish, time: %f'%(end_time - start_time))
print('the new dictionary has %d words'%len(training_term_list))
#print(training_term_interval)

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

	#===============================
	#new method
	#===============================

tr_new_W, feature_interval = facetW(training_W, num_topwords, training_term_interval)
tr_zero_W = W_set_zero(tr_new_W, feature_interval, training_term_interval)

tr_zero_H = nmf_keepW(training_set_normalize, tr_zero_W)
test_zero_H = nmf_keepW(test_set_normalize, tr_zero_W)

tr_zero_H_trans = tr_zero_H.transpose()
test_zero_H_trans = test_zero_H.transpose()

knn(num_neighbors, tr_zero_H_trans, training_categorie, test_zero_H_trans, test_categorie)
