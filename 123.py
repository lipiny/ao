# -*- coding: utf-8 -*-
import io  
import sys  
import urllib.request  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

from ReadTermList import readfile_termlist
from ReadInfoData import readfile_infodata
from sklearn import preprocessing
from nmf import nmf
from nmf_keepW import nmf_keepW
from TopWords import topwords
from BuildTopwordsMatrix import build_topwords_matrix
from nmfSetZero import nmf_set_zero
from knn import knn
from svm import svmclf

#=====================parameter====================
num_topwords = 20
#--number of data to use
trainning_rescale_num = 1000
test_rescale_num = 100
#---NMF
n_components=70
#---knn
num_neighbors=5
#---file name
filename_termList = 'termList'
filename_trainning_info = 'train.info'
filename_test_info = 'test.info'
filename_trainning_data = 'train.data'
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

trainning_set, trainning_doc_num, trainning_title, trainning_categorie = readfile_infodata(trainning_rescale_num, term_num, filename_trainning_info, filename_trainning_data)
test_set, test_doc_num, test_title, test_categorie = readfile_infodata(test_rescale_num, term_num, filename_test_info, filename_test_data)

trainning_set_normalize=preprocessing.normalize(trainning_set, axis=0)
test_set_normalize=preprocessing.normalize(test_set,axis=0)

	#===============================
	#nmf and knn
	#===============================

trainning_W, trainning_H = nmf(trainning_set_normalize, n_components)
test_H = nmf_keepW(test_set_normalize, trainning_W, n_components)

trainning_H_trans = trainning_H.transpose()
test_H_trans = test_H.transpose()

knn(num_neighbors, trainning_H_trans, trainning_categorie, test_H_trans, test_categorie)

	#===============================
	#new method
	#===============================


top_words = topwords(trainning_W, n_components, num_topwords)

trainning_set_normalize_trans = trainning_set_normalize.tocsr()
test_set_normalize_trans = test_set_normalize.tocsr()
trainning_topwords_matrix = build_topwords_matrix(trainning_set_normalize_trans, top_words, trainning_doc_num)
test_topwords_matrix = build_topwords_matrix(test_set_normalize_trans, top_words, test_doc_num)

trainning_topwords_set_normalize=preprocessing.normalize(trainning_topwords_matrix, axis=0)
test_topwords_set_normalize=preprocessing.normalize(test_topwords_matrix, axis=0)

tw_trainning_W, tw_trainning_H = nmf(trainning_topwords_set_normalize, n_components)
tw_test_H = nmf_keepW(test_topwords_set_normalize, tw_trainning_W, n_components)

tw_trainning_H_trans = tw_trainning_H.transpose()
tw_test_H_trans = tw_test_H.transpose()

knn(num_neighbors, tw_trainning_H_trans, trainning_categorie, tw_test_H_trans, test_categorie)


'''
tw_trainning_W, tw_trainning_H = nmf_set_zero(trainning_topwords_set_normalize, n_components)
tw_test_H = nmf_keepW(test_topwords_set_normalize, tw_trainning_W, n_components)

tw_trainning_H_trans = tw_trainning_H.transpose()
tw_test_H_trans = tw_test_H.transpose()

knn(num_neighbors, tw_trainning_H_trans, trainning_categorie, tw_test_H_trans, test_categorie)
'''
