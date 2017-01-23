import numpy as np

def W_set_zero(W, feature_interval, term_interval):
	len_target = 0
	for i in range(0, len(feature_interval)-1):
		len_target = len_target + feature_interval[i]
		
	for i in range(0, len(term_interval)-1):
		judge = np.array([])
		for j in range(0, int(len_target)):
			judge = np.append(judge, [0])
		start_point = 0
		for j in range(0, i):
			start_point = start_point + feature_interval[j]
		for j in range(int(start_point), int(feature_interval[i])):
			judge[j] = 1
		for j in range(term_interval[i], term_interval[i+1]):
			for k in range(0, int(len_target)):
				if(judge[k] == 0):
					W[j][k] = 0
	
	return W
				
				