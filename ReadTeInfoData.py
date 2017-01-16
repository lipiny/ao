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
def readfile_te_infodata(order, begin, end, rescale_num, filename_info, filename_data, training_term_list):

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
	#result indptr, indices, data
	#===============================

	indptr = np.array([0])
	indices = np.array([])
	data = np.array([])

	for i in range(0,art_num):
		index_1 = 0
		index_2 = 0
		judge = 0

		num_words = 0

		indptr_tmp = 0
		indices_tmp = -1
		data_tmp = -1
		
		#---indptr
		while((lines_data[sample_docs[i]][index_2] == ' ') is False):
			index_2 = index_2 + 1
		indptr_tmp = int(lines_data[sample_docs[i]][index_1:index_2])
		num_words = int(indptr_tmp)

		for j in range(0, num_words):
			index_2 = index_2 + 1
			index_1 = index_2
			#---indices
			while((lines_data[sample_docs[i]][index_2] == ':') is False):
				index_2 = index_2 + 1
			#print('in loop %d the index is from %d to %d.'%(i, index_1, index_2))
			indices_tmp=int(lines_data[sample_docs[i]][index_1:index_2])
			for k in range(0, len(training_term_list)):
				if( indices_tmp == training_term_list[k] ):
					judge = 1
					indices = np.append(indices, [k])
					break
			if( judge == 0 ):
				#print('judge = 0')
				indptr_tmp = indptr_tmp - 1
				if( j < num_words -1 ):
					while((lines_data[sample_docs[i]][index_2] == ' ') is False):
						index_2 = index_2 + 1
			elif( judge == 1 ):
				#print('judge = 1')
				index_2 = index_2 + 1
				index_1 = index_2
				if(j < num_words - 1):
					while((lines_data[sample_docs[i]][index_2] == ' ') is False):
						index_2 = index_2 + 1
					data_tmp=int(lines_data[sample_docs[i]][index_1:index_2])
				elif( j == num_words - 1 ):
					data_tmp=int(lines_data[sample_docs[i]][index_1:])
				else:
					print('error in [ReadTeInfoData.py] with j')
					exit()
				data = np.append(data, [data_tmp])
			else:
				print('error in [ReadTeInfoData.py] with judge')
				exit()
			#print('in loop %d the index is from %d to %d.'%(j, index_1, index_2))
			judge = 0
		indptr = np.append(indptr, [indptr[len(indptr)-1] + indptr_tmp])
		#---info data---
		#----finish----

	#===============================
	#compress to a CSC matrix
	#===============================
	term_list_CSC_matrix = csc_matrix((data, indices, indptr), shape=(len(training_term_list), art_num))
	#print(type(term_list_CSC_matrix))
	#term_list_CSR_matrix = term_list_CSC_matrix.tocsr()
	#print(type(term_list_CSR_matrix))
	
	return term_list_CSC_matrix, art_num, title, categorie
