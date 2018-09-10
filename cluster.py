import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection
import sklearn.preprocessing
import sys



def evaluate(model, X, y):
	# compute categorical labels
	classes = list(set(y))
	y_cate = [classes.index(y_i) for y_i in y]

	# predict output
	try:
		y_pred = model.fit_predict(X)
	except AttributeError:
		model.fit(X)
		y_pred = model.predict(X)

	# compute metrics
	metrics = [
		("ari", sklearn.metrics.adjusted_rand_score(y, y_pred)),
		("ami", sklearn.metrics.adjusted_mutual_info_score(y, y_pred))
	]

	for (name, value) in metrics:
		print("%4s = %8.3f" % (name, value))



def create_kmeans(n_clusters):
	return sklearn.cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)



def create_minibatch_kmeans(n_clusters):
	return sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters)



def create_affinity_propagation(n_clusters):
	return sklearn.cluster.AffinityPropagation()



def create_mean_shift(n_clusters):
	return sklearn.cluster.MeanShift(n_jobs=-1)



def create_spectral_clustering(n_clusters):
	return sklearn.cluster.SpectralClustering(n_clusters=n_clusters, n_jobs=-1)



def create_agglomerative_clustering(n_clusters):
	return sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)



def create_dbscan(n_clusters):
	return sklearn.cluster.DBSCAN(n_jobs=-1)



def create_birch(n_clusters):
	return sklearn.cluster.Birch(n_clusters=n_clusters)



def create_gaussian_mixture(n_clusters):
	return sklearn.mixture.GaussianMixture(n_components=n_clusters)



def create_bayesian_gaussian_mixture(n_clusters):
	return sklearn.mixture.BayesianGaussianMixture(n_components=n_clusters)



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: python regress.py [infile] [config]")
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	# load data, extract X and y
	df = pd.read_table(sys.argv[1])
	df_cate = df[config["categorical"]]
	X = df[config["numerical"]]
	y = df[config["output"][0]]

	# determine number of clusters (classes)
	n_clusters = len(list(set(y)))

	# evaluate each clustering method
	methods = [
		("k-means", create_kmeans),
		("mini-batch k-means", create_minibatch_kmeans),
		("affinity propagation", create_affinity_propagation),
		("mean shift", create_mean_shift),
		("spectral clustering", create_spectral_clustering),
		("agglomerative clustering", create_agglomerative_clustering),
		("DBSCAN", create_dbscan),
		("Birch", create_birch),
		("Gaussian mixture model", create_gaussian_mixture),
		("Bayesian Gaussian mixture model", create_bayesian_gaussian_mixture)
	]

	for (name, create_model) in methods:
		print("Evaluating %s..." % (name))
		model = create_model(n_clusters)
		evaluate(model, X, y)
		print
