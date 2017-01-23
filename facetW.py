import numpy as np
from collections import Counter

def facetW(W, n_top_words, term_interval, factor1=3, factor2=0.4):
	n_terms = len(W)
	n_features = len(W[0])
	
	classify = -1
	new_order = np.array([])
	W = W.transpose()
	for i in range(0, n_features):
		topwords_tmp = np.array([])
		topwords = np.array([])
		for j in W[i].argsort()[:-n_top_words - 1:-1]:
			topwords_tmp = np.append(topwords_tmp, [j])
		for j in range(0, len(topwords_tmp)):
			new_topwords_tmp = -1
			for k in range(0, len(term_interval)-1):
				if(topwords_tmp[j] >= term_interval[k] and topwords_tmp[j] < term_interval[k+1]):
					new_topwords_tmp = k
					break;
			if(new_topwords_tmp != -1):
				topwords = np.append(topwords, [new_topwords_tmp])
		if(len(topwords)==0 or n_top_words/len(topwords) > factor1):
			classify = len(term_interval)-1
		else:
			n_first_rank = Counter(topwords).most_common(1)[0][1]
			if(n_first_rank/len(topwords) >= factor2):
				classify = Counter(topwords).most_common(1)[0][0]
			else:
				classify = len(term_interval)-1
		new_order = np.append(new_order, [classify])
	
	print(new_order)
	
	new_order_list = np.unique(new_order)
	print(new_order_list)
	
	count_sp = 0
	exchange_interval = 0
	feature_interval = np.array([])
	for i in new_order_list:
		count_1 = 0
		while(Counter(new_order).most_common(len(term_interval))[count_1][0] != i):
			count_1 = count_1 + 1
		num_exchange = Counter(new_order).most_common(len(term_interval))[count_1][1]
		
		count_2 = exchange_interval
		for j in range(exchange_interval, exchange_interval + num_exchange):
			while(new_order[count_2] != i):
				count_2 = count_2 + 1
			W[[j,count_2]] = W[[count_2,j]]
		exchange_interval = exchange_interval + num_exchange
		while(count_sp < i):
			count_sp = count_sp + 1
			feature_interval = np.append(feature_interval, [0])
		feature_interval = np.append(feature_interval, [num_exchange])
		count_sp = count_sp + 1
	W = W.transpose()

	return W, feature_interval
