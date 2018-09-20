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



if __name__ == '__main__':

	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset, \
		subset of genes, or a random set')
	parser.add_argument('--dataset', help='dataset to be used', type=str, required=True)
	parser.add_argument('--gene_list', help='list of genes in dataset (same order as dataset)', \
		type=str, required=True)

	args = parser.parse_args()

	# Check arguments are correct
	check_args(args)

	# load the data
	print('loading genetic data...')
	gtex_gct_flt = np.load(args.dataset)
	total_gene_list = np.load(args.gene_list)
	data = load_data(args.sample_json, gtex_gct_flt)

	# load interaction data, if passed
	if args.interaction_genes:
		interaction_genes = np.load(args.interaction_genes)

		# ensure only genes in interaction_genes are contained within the dataset
		interaction_genes = [g for g in interaction_genes if g in total_gene_list]
	else:
		interaction_genes = None

	if args.interaction_list:
		interaction_list = np.load(args.interaction_list)
		original_interaction_genes = np.load(args.interaction_genes)

		# find missing genes, then delete them from interaction list
		missing = [g for g in original_interaction_genes if g not in total_gene_list]
		for g in missing:
			locs = np.where(interaction_list==g)
			interaction_list = np.delete(interaction_list, locs[0], axis=0)

		interaction_genes = list(np.unique(interaction_list))
	else:
		interaction_list = None

	# ensure the dataset and gene list match dimensions
	#assert gtex_gct_flt.shape[0] not total_gene_list.shape[0], "dataset does not match gene list."
	if gtex_gct_flt.shape[0] != total_gene_list.shape[0]:
		print('dataset does not match gene list.')
		sys.exit(1)

	config = json.load(open(args.config))

	# RUN SUBSET CLASSIFICATION
	# read subset file, if provided
	if args.subset_list and not args.random_test:
		subsets = read_subset_file(args.subset_list)

		tot_genes = []
		missing_genes = []

		print('checking for valid genes...')
		for s in subsets:
			genes = []
			for g in subsets[s]:
				if g not in tot_genes:
					tot_genes.append(g)
				if g in total_gene_list:
					genes.append(g)
				else:
					if g not in missing_genes:
						missing_genes.append(g)
			subsets[s] = genes
					#print('missing gene ' + str(g))
		print('missing ' + str(len(missing_genes)) + '/' + str(len(tot_genes)) + ' genes' + ' or ' \
			 + str(int((float(len(missing_genes)) / len(tot_genes)) * 100.0)) + '% of genes')

		if args.set:
			sub = {}
			sub[args.set.upper()] = subsets[args.set.upper()]
			subsets = sub

		subset_classification(data, total_gene_list, config, subsets, args.out_file, kfold_val=10)


	#RUN RANDOM CLASSIFICATION
	# if random is selectioned, run random
	if args.random_test:
		if args.num_random_genes:
			random_classification(data, total_gene_list, config, args.num_random_genes, args.rand_iters, args.out_file, kfold_val=10)
		elif args.subset_list:
			# get the number of genes for each subset
			num = []
			subsets = read_subset_file(args.subset_list)
			for s in subsets:
				genes = []
				for g in subsets[s]:
					if g in total_gene_list:
						genes.append(g)
				subsets[s] = genes

			for k in subsets:
				num.append(len(subsets[k]))
			num.sort()
			random_classification(data, total_gene_list, config, num, args.rand_iters, \
									args.out_file, kfold_val=3, interaction_genes=interaction_genes, \
									interaction_list=interaction_list)


	#RUN FULL_CLASSIFICATION
	# if not subset test and random test, run classifier on all 56k genes
	if not args.random_test and not args.subset_list:
		full_classification(data, total_gene_list, config, args.out_file)
