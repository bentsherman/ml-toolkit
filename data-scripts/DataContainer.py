#
# FILE: dataset.py
# USE:  Creates object that holds data and labels for gene specified dataset
#

import numpy as np
import random
import sys

class data_t(object):
	def __init__(self, data, labels):
		self.labels = labels
		self.data = data
		self.num_examples = data.shape[0]

	def next_batch(self, batch_size, index):
		idx = index * batch_size
		n_idx = index * batch_size + batch_size
		return self.data[idx:n_idx, :], self.labels[idx:n_idx, :]

class DataContainer:
	def __init__(self, data, total_gene_list=None):
		self.num_classes = len(data)
		self.label_names_ordered = []
		self.class_counts = {}
		self.train, self.test = self.split_set(data, total_gene_list, sub_gene_list, train_split, test_split)


	#
	# USAGE:
	#	TODO: What is this use @Colin
	def shuffle(self):
		idxs = np.arange(self.train.data.shape[0])
		idxs = np.random.shuffle(idxs)
		self.train.data = np.squeeze(self.train.data[idxs])
		self.train.labels = np.squeeze(self.train.labels[idxs])



	#
	# USAGE:
	#       partition dataset into train/test sets
	#
	def partition(self, data, percent_samples=1.0, train_split=70, test_split=30):
