import math
import numpy as np

def melt(params, q_data):
	# dataArray: [ array([[],[],..])] Shape: (3633, 200)
	N = int(math.ceil(float(len(q_data)) / float(params.batch_size))) #number of batches
	q_data = q_data.T  # Shape: (200,3633) #transpose it.
	seq_num = q_data.shape[1]
	target_list = []
	count = 0
	element_count = 0
	for idx in range(N):
		inds = np.arange(idx * params.batch_size, (idx + 1) * params.batch_size) #arange is an array version of Python's range. so we get indices for the items we're working on.
		q_one_seq = q_data.take(inds, axis=1, mode='wrap') #and we pull them out.
		input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
		target = input_q
		if (idx + 1) * params.batch_size > seq_num:
			real_batch_size = seq_num - idx * params.batch_size
			target = input_q[:, :real_batch_size]
			count += real_batch_size
		else:
			count += params.batch_size
		#here, target is 200*50, where 200 is problems and 50 is students (batched at 50, hence the 50)
		#hence, target[0] is a length-50 array of answers to problem number 1, for the relevant 50 students. answers as keyed below.
		target = target.reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0
		#this reshape just flattens it. It's now a length-10K array, in order. 50 answers for problem 1, then 50 for problem 2, etc.
		nopadding_index = np.flatnonzero(target != 0)
		nopadding_index = nopadding_index.tolist() #now we pull out the blanks.
		target_nopadding = target[nopadding_index]
		target_list.append(target_nopadding)
	assert count == seq_num
	all_target = np.concatenate(target_list, axis=0)
	import pandas as pd
	pd.DataFrame(all_target).to_csv("all_map.csv")
	return all_target

def load_data(path):
	with open(path , 'r') as f_data:
		q_data = []
		for lineID, line in enumerate(f_data):
			line = line.strip( )
			# lineID starts from 0
			Q = line.split(",")
			if len( Q[len(Q)-1] ) == 0:
				Q = Q[:-1]
			#print(len(Q))
			n_split = 1
			#print('len(Q):',len(Q))
			if len(Q) > 200:
				n_split = math.floor(len(Q) / 200)
				if len(Q) % 200:
					n_split = n_split + 1
			#print('n_split:',n_split)
			for k in range(n_split):
				question_sequence = []
				if k == n_split - 1:
					endINdex  = len(Q)
				else:
					endINdex = (k+1) * 200
				for i in range(k * 200, endINdex):
					if len(Q[i]) > 0 :
						# int(A[i]) is in {0,1}
						question_sequence.append(Q[i])
					else:
						print(Q[i])
				#print('instance:-->', len(instance),instance)
				q_data.append(question_sequence)
	### data: [[],[],[],...] <-- set_max_seqlen is used
	### convert data into ndarrays for better speed during training
	q_dataArray = np.zeros((len(q_data), 200), dtype='object')
	for j in range(len(q_data)):
		dat = q_data[j]
		q_dataArray[j, :len(dat)] = dat
	# dataArray: [ array([[],[],..])] Shape: (3633, 200)
	return q_dataArray