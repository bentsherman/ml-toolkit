import json
import math
import pandas as pd
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sys



def test_model(reg, X, y):
	scores = sklearn.model_selection.cross_val_score(reg, X, y, cv=5)
	print "r = %8.3f, r^2 = %8.3f +/- %.3f" % (math.sqrt(max(0, scores.mean())), scores.mean(), scores.std())
	print



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "usage: python regress.py [infile] [config]"
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	df = pd.read_csv(sys.argv[1], sep="\t")
	df_nume = df.drop(config["categorical"], axis=1)
	df_cate = df[config["categorical"]]

	X = df_nume.drop(config["prediction"], axis=1)
	y = df_nume[config["prediction"][0]]

	X.iloc[:] = sklearn.preprocessing.scale(X)

	print "Evaluating linear regression..."
	reg = sklearn.linear_model.LinearRegression()
	test_model(reg, X, y)

	print "Evaluating ridge regression..."
	reg = sklearn.linear_model.Ridge()
	test_model(reg, X, y)

	print "Evaluating lasso regression..."
	reg = sklearn.linear_model.Lasso()
	test_model(reg, X, y)

	print "Evaluating elastic net regression..."
	reg = sklearn.linear_model.ElasticNet()
	test_model(reg, X, y)
	print
	print "Evaluating Bayesian ridge regression..."
	reg = sklearn.linear_model.BayesianRidge()
	test_model(reg, X, y)

	print "Evaluating SGD..."
	reg = sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
	test_model(reg, X, y)

	print "Evaluating Polynomial regression..."
	X_poly = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(X)
	reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
	test_model(reg, X_poly, y)

	print "Evaluating Kernel ridge regression..."
	reg = sklearn.kernel_ridge.KernelRidge()
	test_model(reg, X, y)

	print "Evaluating SVM (linear kernel)..."
	reg = sklearn.svm.SVR(kernel="linear")
	test_model(reg, X, y)

	print "Evaluating SVM (polynomial kernel)..."
	reg = sklearn.svm.SVR(kernel="poly")
	test_model(reg, X, y)

	print "Evaluating SVM (RBF kernel)..."
	reg = sklearn.svm.SVR(kernel="rbf")
	test_model(reg, X, y)

	print "Evaluating MLP..."
	reg = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(30,), max_iter=500)
	test_model(reg, X, y)