# -*- coding: utf-8 -*-
import numpy as np
import os
import io
'''
import sys  
import urllib.request  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
'''
def readfile_termlist(filename):

	#===============================
	#check file existance and read data
	#===============================

	if (os.path.exists(filename)) is False:
		print('file %s not exist'%filename)
		exit()
	fileRead = io.open(filename, 'r', encoding='utf8')
	lines = fileRead.readlines()
	fileRead.close()

	#===============================
	#store data in one(two) array　and return
	#===============================


	num_term = len(lines)
	term=['' for i in range(0,num_term)]
	for i in range(0,num_term):
		index_1 = 0
		while((lines[i][index_1] == '\t') is False):
			index_1 = index_1 + 1
		index_2 = index_1 + 1
		while((lines[i][index_2] == '\t') is False):
			index_2 = index_2 + 1
		term[i]=lines[i][index_1+1:index_2]

	return num_term, term
