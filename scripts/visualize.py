import json
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.decomposition
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



def plot_pca(df, X, y):
	# compute PCA
	pca = sklearn.decomposition.PCA()
	pca.fit(X)

	# plot explained variance of each principal component
	plt.plot(pca.explained_variance_ratio_)
	plt.title("Explained Variance of Principal Components")
	plt.xlabel("Component")
	plt.ylabel("Explained Variance Ratio")
	plt.xlim(0, len(pca.explained_variance_ratio_) - 1)
	plt.show()

	# plot projection of dataset onto two principal axes
	X_proj = pca.transform(X)
	idx = [0, 1]

	if y.dtype == "object":
		classes = list(set(y))

		for c in classes:
			mask = (y == c)
			plt.scatter(X_proj[mask, idx[0]], X_proj[mask, idx[1]], label=c)
			plt.xlabel("Principal Component %d" % (idx[0]))
			plt.ylabel("Principal Component %d" % (idx[1]))

		plt.legend()
	else:
		paths = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
		plt.colorbar(paths)

	plt.show()



def plot_tsne_2d(df, X, y):
	X_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(X)

	plt.axis("off")

	if y.dtype == "object":
		classes = list(set(y))

		for c in classes:
			idx = (y == c)
			plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=c)

		plt.legend()
	else:
		paths = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
		plt.colorbar(paths)

	plt.show()



def plot_tsne_3d(df, X, y):
	X_tsne = sklearn.manifold.TSNE(n_components=3).fit_transform(X)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	if y.dtype == "object":
		classes = list(set(y))

		for c in classes:
			idx = (y == c)
			ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], X_tsne[idx, 2], label=c)

		plt.legend()
	else:
		paths = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y)
		plt.colorbar(paths)

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

	methods = [
		("heatmap", plot_heatmap),
		("feature distributions", plot_dist_input),
		("output distribution", plot_dist_output),
		("contingency tables", plot_contingency_tables),
		("correlation heatmap", plot_correlation_heatmap),
		("correlation clustermap", plot_correlation_clustermap),
		("pairwise feature distributions", plot_pairwise),
		("principal component analysis", plot_pca),
		("2-D t-SNE", plot_tsne_2d),
		("3-D t-SNE", plot_tsne_3d)
	]

	for (name, method) in methods:
		print("Plotting %s..." % (name))
		method(df, X, y)
