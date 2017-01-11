# -*- coding: utf-8 -*-
import numpy as np
import os
import io
import random
#from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix


#order = 'all'			parameter:none
#order = 'continuous'	parameter:begin, rescale_num
#order = 'backorder'	parameter:rescale_num
#order = 'random'		parameter:begin, end, rescale_num
def readfile_tr_infodata(order, begin, end, rescale_num, filename_info, filename_data):

	#---data read---
	#----begin----
	#===============================
	#check file existance and read data
	#===============================

	if (os.path.exists(filename_info)) is False:
		print('file %s not exist'%filename_info)
		exit()
	fileRead = io.open(filename_info, 'r', encoding='utf8')
	lines_info = fileRead.readlines()  
	fileRead.close()

	#===============================
	#store titles and categories in two array　and return
	#===============================

	total_art_num = len(lines_info)

	if ( order == 'all' ):
		art_num = total_art_num
		sample_docs = []
		for i in range(0, art_num):
			sample_docs.append(i)
	elif ( order == 'continuous' ):
		art_num = rescale_num
		sample_docs = []
		for i in range(begin, begin + rescale_num + 1):
			sample_docs.append(i)
	elif ( order == 'backorder' ):
		art_num = rescale_num
		sample_docs = []
		for i in range(total_art_num - rescale_num, total_art_num):
			sample_docs.append(i)
	# in mode random, the article No.end is not included.
	elif ( order == 'random' ):
		art_num = rescale_num
		if ( end - begin + 1 >= rescale_num ):
			sample_docs=random.sample(range(begin,end+1),rescale_num)
	else:
		print("error: wrong mode setting in <readfile_tr_infodata>!")
		exit();
	'''
	for i in range(0,art_num):
		print(sample_docs[i])
	'''

	title=['' for i in range(0,art_num)]
	categorie=['' for i in range(0,art_num)]
	
	for i in range(0,art_num):
		count = 0
		while((lines_info[sample_docs[i]][count] == '\t') is False):
			count = count + 1
		title[i]=lines_info[sample_docs[i]][0:count]
		categorie[i]=lines_info[sample_docs[i]][count+1:]

	#---info read---
	#----finish----



	#---data read---
	#----begin----
	#===============================
	#check file existance and read data
	#===============================
	if (os.path.exists(filename_data)) is False:
		print('file %s not exist'%filename_data)
		exit()
	fileRead = io.open(filename_data, 'r', encoding='utf8')
	lines_data = fileRead.readlines()  
	fileRead.close()

	#===============================
	#CSR format
	#result indptr
	#===============================

	#---'+1' because indptr[0]=0---
	indptr = np.zeros((art_num+1))
	#indptr[0] = int(0)

	for i in range(0,art_num):
		count = 0
		while((lines_data[sample_docs[i]][count] == ' ') is False):
			count = count + 1
		indptr[i+1]=int(lines_data[sample_docs[i]][0:count])+int(indptr[i])
	#---indptr is a float array and I don't know how to convert---
	#---so when you use it please add int()---
	#---ok I know the reason. This is because I am using numpy array.

	#===============================
	#CSR format
	#result indices, data
	#===============================
	indices = np.zeros((int(indptr[art_num])))
	data = np.zeros((int(indptr[art_num])))

	for i in range(0,art_num):
		index_1 = 0
		index_2 = 0
		count = 0
		while((lines_data[sample_docs[i]][index_2] == ' ') is False):
			index_2 = index_2 + 1
		index_1 = index_2
		#---from element 0 to the last 2nd one---
		for j in range(0, int(indptr[i+1])-int(indptr[i])-1):
			while((lines_data[sample_docs[i]][index_2] == ':') is False):
				index_2 = index_2 + 1
			indices[int(indptr[i])+count]=int(lines_data[sample_docs[i]][index_1+1:index_2])
			index_1 = index_2
			while((lines_data[sample_docs[i]][index_2] == ' ') is False):
				index_2 = index_2 + 1
			data[int(indptr[i])+count]=int(lines_data[sample_docs[i]][index_1+1:index_2])
			index_1 = index_2
			count = count + 1
			#print(data[int(indptr[i])+count])
		#---deal with the last one---
		while((lines_data[sample_docs[i]][index_2] == ':') is False):
				index_2 = index_2 + 1
		indices[int(indptr[i])+count]=int(lines_data[sample_docs[i]][index_1+1:index_2])
		data[int(indptr[i])+count]=int(lines_data[sample_docs[i]][index_2+1:])

	#---info data---
	#----finish----

	#generate the dictionary
	training_term_list = np.unique(indices)
	term_list_num = len(training_term_list)
	
	#adjust the indices
	new_indices = np.zeros((int(indptr[art_num])))
	for i in range(0,art_num):
		#count = 0
		for j in range(0, int(indptr[i+1])-int(indptr[i])):
			count = 0
			while ( indices[j] != training_term_list[count] ):
				count = count + 1
			new_indices[int(indptr[i])+j] = count

	#===============================
	#compress to a CSC matrix
	#===============================
	term_list_CSC_matrix = csc_matrix((data, new_indices, indptr), shape=(term_list_num, art_num))
	#print(type(term_list_CSC_matrix))
	#term_list_CSR_matrix = term_list_CSC_matrix.tocsr()
	#print(type(term_list_CSR_matrix))
	
	return term_list_CSC_matrix, training_term_list, art_num, title, categorie
