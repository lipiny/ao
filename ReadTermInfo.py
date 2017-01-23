# -*- coding: utf-8 -*-
import numpy as np
import os
import io
'''
import sys  
import urllib.request  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
'''
def readfile_terminfo(filename):

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
	#store data in one array　and return
	#===============================

	term_interval = np.array([])
	for i in range(0,len(lines)):
		term_interval=np.append(term_interval, [int(lines[i])])

	return term_interval
