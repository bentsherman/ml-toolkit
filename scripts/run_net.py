#/usr/bin/python

'''
	This script can be used to run a specified dataset, a specified subset of genes,
	or a specified number of random genes for classification.

	It is required to have a numpy array containing column-wise samples and rows of
	genes. Additionally, it is required to have a numpy vector of genes that are contained
	in the dataset (in the same exact order).


	Protypes:
	- random_classification(data, total_gene_list, config, num_genes, iters, out_file, kfold_val)
	- subset_classification(data, total_gene_list, config, subsets, out_file, kfold_val=1)
	- full_classification(data, total_gene_list, config, out_file, kfold_val=1)

	Todo:
		-modularize functions in main, main should be small, logic needs to be in functions

'''

import numpy as np
import sys, argparse
import os
import json
import time

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from models.mlp import MLP
from models.cnn import CNN
from models.pointnet import PointNet
from utils.DataContainer import DataContainer as DC




if __name__ == '__main__':

	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset, \
		subset of genes, or a random set')
	parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
	parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
	parser.add_argument('--net', help='which type of network to run (mlp/cnn)', type=str, required=False, \
									choices=['mlp', 'cnn', 'pc'], default='mlp')

	args = parser.parse_args()

	print('loading numpy data...')
	data = np.load(args.dataset)
	labels = np.load(args.labels)

	print('converting to DataContainer format...')
	dc = DC(data=data, labels=labels)

	# trim distance matrices for experiments
	#dc.train.data = dc.train.data[:,:20,:20]
	#dc.test.data = dc.test.data[:,:20,:20]

	if args.net == 'mlp':
		# dc.train.data = dc.train.data.reshape(dc.train.data.shape[0], -1)
		# dc.test.data = dc.test.data.reshape(dc.test.data.shape[0], -1)
		# triu_i = np.triu_indices(dc.train.data.shape[-1], k=1)
		# dc.train.data = dc.train.data[:, triu_i[0], triu_i[1]]
		# dc.test.data = dc.test.data[:, triu_i[0], triu_i[1]]
		net = MLP(epochs=40,
				  h_units=[512, 128, 32],
				  batch_size=64,
				  n_input=dc.train.data.shape[-1],
				  verbose=1)
	
	if args.net == 'cnn':
		net = CNN(epochs=20,
				  batch_size=64,
				  n_input=dc.train.data.shape[-1],
				  verbose=1)

	if args.net == 'pc':
		net = PointNet(epochs=5,
					   batch_size=64,
					   n_points=25,
					   n_input=3, 
					   verbose=1)


	print('train shape: ' + str(dc.train.data.shape))
	print('test shape: ' + str(dc.test.data.shape))

	acc = net.run(dc)

	print('final accuracy: ' + str(acc))

