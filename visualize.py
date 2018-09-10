import json
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold
import sklearn.preprocessing
import sys



def rotate_xticklabels(angle):
	for tick in plt.gca().get_xticklabels():
		tick.set_horizontalalignment("right")
		tick.set_rotation(angle)



def contingency_table(x, y, data):
	u = list(set(data[x]).union(set(data[y])))
	T = pd.DataFrame(np.zeros((len(u), len(u))), index=u, columns=u, dtype=np.int32)

	for k in range(len(data)):
		i = u.index(data[x][k])
		j = u.index(data[y][k])

		T.iloc[i, j] += 1

	return T



def plot_heatmap(df, X, y):
	fig, ax = plt.subplots(figsize=(X.shape[1] / 3, 6))
	sns.heatmap(X, xticklabels=1)
	rotate_xticklabels(45)
	plt.show()



def plot_dist_input(df, X, y):
	fig, ax = plt.subplots(figsize=(X.shape[1] / 2, 6))
	sns.violinplot(data=X, bw=0.2, cut=1, linewidth=1)
	rotate_xticklabels(45)
	plt.show()



def plot_dist_output(df, X, y):
	if y.dtype == "object":
		sns.countplot(y)
	else:
		sns.distplot(y)

	plt.show()



def plot_contingency_tables(df, X, y):
	for i in range(0, len(config["categorical"]), 2):
		x = config["categorical"][i]
		y = config["categorical"][i + 1]
		T = contingency_table(x, y, df)

		ax = sns.heatmap(T, annot=True, fmt="d")
		ax.set_ylabel(x)
		ax.set_xlabel(y)
		rotate_xticklabels(45)
		plt.show()



def plot_correlation_heatmap(df, X, y):
	fig, ax = plt.subplots(figsize=(X.shape[1] / 3, X.shape[1] / 3))
	sns.heatmap(X.corr(), xticklabels=1, yticklabels=1)
	rotate_xticklabels(45)
	plt.show()



def plot_correlation_clustermap(df, X, y):
	# fig, ax = plt.subplots(figsize=(X.shape[1] / 3, X.shape[1] / 3))
	sns.clustermap(X.corr(), xticklabels=1, yticklabels=1)
	rotate_xticklabels(45)
	plt.show()



def plot_pairwise(df, X, y):
	g = sns.PairGrid(X, diag_sharey=False)
	g.map_lower(plt.scatter, s=2)
	g.map_diag(sns.kdeplot, lw=2, legend=False)
	plt.show()



def plot_tsne_2d(df, X, y):
	X_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(X)

	plt.axis("off")

	if y.dtype == "object":
		classes = list(set(y))

		for c in classes:
			idx = (y == c)
			plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=3, label=c)

		plt.legend()
	else:
		plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=3)
		# TODO: colorbar?

	plt.show()



def plot_tsne_3d(df, X, y):
	X_tsne = sklearn.manifold.TSNE(n_components=3).fit_transform(X)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	if y.dtype == "object":
		classes = list(set(y))

		for c in classes:
			idx = (y == c)
			ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], X_tsne[idx, 2], s=3, label=c)

		plt.legend()
	else:
		ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, s=3)

	plt.show()



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: python visualize.py [infile] [config]")
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	df = pd.read_table(sys.argv[1])
	df_cate = df[config["categorical"]]
	X = df[config["numerical"]]
	y = df[config["output"][0]]

	# apply standard scaler
	X = pd.DataFrame(sklearn.preprocessing.scale(X), X.index, X.columns)

	# remove samples with high-variance output
	output_sd = "%s_SD" % config["output"][0]
	if output_sd in df.columns:
		mask = df[output_sd] < 1.5
		X = X[mask]
		y = y[mask]

	methods = [
		("heatmap", plot_heatmap),
		("feature distributions", plot_dist_input),
		("output distribution", plot_dist_output),
		("contingency tables", plot_contingency_tables),
		("correlation heatmap", plot_correlation_heatmap),
		("correlation clustermap", plot_correlation_clustermap),
		("pairwise feature distributions", plot_pairwise),
		("2-D t-SNE embedding", plot_tsne_2d),
		("3-D t-SNE embedding", plot_tsne_3d)
	]

	for (name, method) in methods:
		print("Plotting %s..." % (name))
		method(df, X, y)
