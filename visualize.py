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



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "usage: python visualize.py [infile] [config]"
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	df = pd.read_csv(sys.argv[1], sep="\t")
	df_nume = df.drop(config["categorical"], axis=1)
	df_cate = df[config["categorical"]]

	X = df_nume.drop(config["prediction"], axis=1)
	y = df_nume[config["prediction"][0]]

	X.iloc[:] = sklearn.preprocessing.scale(X)

	print "Creating heatmap..."
	sns.heatmap(X)
	rotate_xticklabels(45)
	plt.show()

	print "Creating violin plot of features..."
	sns.violinplot(data=X, bw=0.2, cut=1, linewidth=1)
	rotate_xticklabels(45)
	plt.show()

	print "Creating distribution plot of output..."
	sns.distplot(y)
	plt.show()

	print "Creating transition matrices..."

	for i in xrange(0, len(config["categorical"]), 2):
		label1 = config["categorical"][i]
		label2 = config["categorical"][i + 1]
		u1 = np.unique(df_cate[label1])
		u2 = np.unique(df_cate[label2])
		u = list(set(u1).union(set(u2)))

		T = pd.DataFrame(np.zeros((len(u), len(u))), index=u, columns=u, dtype=np.int32)

		for j in xrange(len(df)):
			i1 = u.index(df[label1][j])
			j1 = u.index(df[label2][j])

			T.iloc[i1, j1] += 1

		ax = sns.heatmap(T)
		ax.set_title(label1.split()[0])
		ax.set_ylabel(label1.split()[-1])
		ax.set_xlabel(label2.split()[-1])
		rotate_xticklabels(45)
		plt.show()

	print "Creating correlation heatmap..."
	corr = X.corr()
	sns.heatmap(corr)
	rotate_xticklabels(45)
	plt.show()

	print "Creating correlation clustermap..."
	sns.clustermap(corr)
	rotate_xticklabels(45)
	plt.show()

	# print "Creating pairwise scatter plots..."
	# g = sns.PairGrid(X, diag_sharey=False)
	# g.map_lower(sns.kdeplot, cmap="Blues_d")
	# g.map_upper(plt.scatter)
	# g.map_diag(sns.kdeplot, lw=3, legend=False)
	# plt.show()

	print "Creating 2-D t-SNE visualization..."
	X_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(X).T
	plt.scatter(X_tsne[0], X_tsne[1], c=y, s=2)
	plt.show()

	# print "Creating 3-D t-SNE visualization..."
	# X_tsne = sklearn.manifold.TSNE(n_components=3).fit_transform(X).T
	# density = scipy.stats.gaussian_kde(X_tsne)(X_tsne)
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection="3d")
	# ax.scatter(X_tsne[0], X_tsne[1], X_tsne[2], c=density, s=2)
	# plt.show()
