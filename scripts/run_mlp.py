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
from utils.DataContainer import DataContainer as DC




if __name__ == '__main__':

	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset, \
		subset of genes, or a random set')
	parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
	parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)

	args = parser.parse_args()

	print('loading numpy data...')
	data = np.load(args.dataset)
	labels = np.load(args.labels)

	print('converting to DataContainer format...')
	dc = DC(data=data, labels=labels)

	print dc.train.data.shape
	print dc.train.labels.shape
