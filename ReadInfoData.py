# -*- coding: utf-8 -*-
import numpy as np
import os
import io
import random
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

def readfile_infodata(rescale_num, term_num, filename_info, filename_data):

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
	#store data in two array　and return
	#===============================

	arc_num = rescale_num

	random_docs=random.sample(range(0,len(lines_info)),rescale_num)
	
	title=['' for i in range(0,arc_num)]
	categorie=['' for i in range(0,arc_num)]
	
	for i in range(0,arc_num):
		count = 0
		while((lines_info[random_docs[i]][count] == '\t') is False):
			count = count + 1
		title[i]=lines_info[random_docs[i]][0:count]
		categorie[i]=lines_info[random_docs[i]][count+1:]
		
		
	#---info read finish---
	
	
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

	arc_num = rescale_num
	#---'+1' because indptr[0]=0---
	indptr = np.zeros((arc_num+1))
	#indptr[0] = int(0)

	for i in range(0,arc_num):
		count = 0
		while((lines_data[random_docs[i]][count] == ' ') is False):
			count = count + 1
		indptr[i+1]=int(lines_data[random_docs[i]][0:count])+int(indptr[i])
	#---indptr is a float array and I don't know how to convert---
	#---so when you use it please add int()---

	#===============================
	#CSR format
	#result indices, data
	#===============================
	indices = np.zeros((int(indptr[arc_num])))
	data = np.zeros((int(indptr[arc_num])))

	for i in range(0,arc_num):
		index_1 = 0
		index_2 = 0
		count = 0
		while((lines_data[random_docs[i]][index_2] == ' ') is False):
			index_2 = index_2 + 1
		index_1 = index_2
		#---from element 0 to the last two---
		for j in range(0, int(indptr[i+1])-int(indptr[i])-1):
			while((lines_data[random_docs[i]][index_2] == ':') is False):
				index_2 = index_2 + 1
			indices[int(indptr[i])+count]=int(lines_data[random_docs[i]][index_1+1:index_2])
			index_1 = index_2
			while((lines_data[random_docs[i]][index_2] == ' ') is False):
				index_2 = index_2 + 1
			data[int(indptr[i])+count]=int(lines_data[random_docs[i]][index_1+1:index_2])
			index_1 = index_2
			count = count + 1
			#print(data[int(indptr[i])+count])
		#---deal with the last one---
		while((lines_data[random_docs[i]][index_2] == ':') is False):
				index_2 = index_2 + 1
		indices[int(indptr[i])+count]=int(lines_data[random_docs[i]][index_1+1:index_2])
		data[int(indptr[i])+count]=int(lines_data[random_docs[i]][index_2+1:])

	#===============================
	#compress to a CSR matrix -> CSC
	#===============================
	term_list_CSC_matrix = csc_matrix((data, indices, indptr), shape=(term_num, arc_num))
	#print(type(term_list_CSC_matrix))
	#term_list_CSR_matrix = term_list_CSC_matrix.tocsr()
	#print(type(term_list_CSR_matrix))
	
	return term_list_CSC_matrix, arc_num, title, categorie
