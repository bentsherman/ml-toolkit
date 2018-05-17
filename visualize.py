import json
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.manifold
import sklearn.preprocessing
import sys



def rotate_xticklabels(angle):
	for tick in plt.gca().get_xticklabels():
		tick.set_horizontalalignment("right")
		tick.set_rotation(angle)



def transition_matrix(x, y, data):
	u = list(set(data[x]).union(set(data[y])))
	T = pd.DataFrame(np.zeros((len(u), len(u))), index=u, columns=u, dtype=np.int32)

	for k in xrange(len(data)):
		i = u.index(data[x][k])
		j = u.index(data[y][k])

		T.iloc[i, j] += 1

	return T



def plot_heatmap(df, X, y):
	print "Plotting heatmap..."
	sns.heatmap(X)
	rotate_xticklabels(45)
	plt.show()



def plot_dist_input(df, X, y):
	print "Plotting distributions of features..."
	sns.violinplot(data=X, bw=0.2, cut=1, linewidth=1)
	rotate_xticklabels(45)
	plt.show()



def plot_dist_output(df, X, y):
	print "Plotting distribution of output..."
	sns.distplot(y)
	plt.show()



def plot_transitions(df, X, y):
	print "Plotting transition matrices..."
	for i in xrange(0, len(config["categorical"]), 2):
		x = config["categorical"][i]
		y = config["categorical"][i + 1]
		T = transition_matrix(x, y, df)

		ax = sns.heatmap(T)
		ax.set_title(x.split("_")[0])
		ax.set_ylabel(x.split("_")[-1])
		ax.set_xlabel(y.split("_")[-1])
		rotate_xticklabels(45)
		plt.show()



def plot_correlation_heatmap(df, X, y):
	print "Plotting correlation heatmap..."
	corr = X.corr()
	sns.heatmap(corr)
	rotate_xticklabels(45)
	plt.show()



def plot_correlation_clustermap(df, X, y):
	print "Plotting correlation clustermap..."
	corr = X.corr()
	sns.clustermap(corr)
	rotate_xticklabels(45)
	plt.show()



def plot_pairwise(df, X, y):
	print "Plotting pairwise distributions..."
	g = sns.PairGrid(X, diag_sharey=False)
	g.map_lower(plt.scatter, s=2)
	g.map_diag(sns.kdeplot, lw=2, legend=False)
	plt.show()



def plot_2d(df, X, y):
	print "Plotting 2-D..."
	plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=2)
	plt.show()



def plot_3d(df, X, y):
	print "Plotting 3-D..."
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y, s=2)
	plt.show()



def plot_tsne_2d(df, X, y):
	print "Plotting 2-D t-SNE..."
	X_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(X).T
	plt.scatter(X_tsne[0], X_tsne[1], c=y, s=2)
	plt.show()



def plot_tsne_3d(df, X, y):
	print "Plotting 3-D t-SNE..."
	X_tsne = sklearn.manifold.TSNE(n_components=3).fit_transform(X).T
	density = scipy.stats.gaussian_kde(X_tsne)(X_tsne)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(X_tsne[0], X_tsne[1], X_tsne[2], c=density, s=2)
	plt.show()



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "usage: python visualize.py [infile] [config]"
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	df = pd.read_csv(sys.argv[1], sep="\t")
	df_cate = df[config["categorical"]]
	X = df[config["numerical"]]
	y = df[config["output"][0]]

	# apply standard scaler
	X = pd.DataFrame(sklearn.preprocessing.scale(X), X.index, X.columns)

	# remove samples with high-variance output
	mask = df[config["output"][1]] < 1.5
	X = X[mask]
	y = y[mask]

	methods = [
		plot_heatmap,
		plot_dist_input,
		plot_dist_output,
		plot_transitions,
		plot_correlation_heatmap,
		plot_correlation_clustermap,
		plot_pairwise,
		plot_2d,
		plot_3d,
		plot_tsne_2d,
		plot_tsne_3d
	]

	for method in methods:
		method(df, X, y)
