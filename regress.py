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



def evaluate(reg, X, y):
	scores = sklearn.model_selection.cross_val_score(reg, X, y, cv=5)
	print "r = %8.3f, r^2 = %8.3f +/- %.3f" % (math.sqrt(max(0, scores.mean())), scores.mean(), scores.std())
	print



def evaluate_linear(X, y):
	print "Evaluating linear regression..."
	reg = sklearn.linear_model.LinearRegression()
	evaluate(reg, X, y)



def evaluate_ridge(X, y):
	print "Evaluating ridge regression..."
	reg = sklearn.linear_model.Ridge()
	evaluate(reg, X, y)



def evaluate_lasso(X, y):
	print "Evaluating lasso regression..."
	reg = sklearn.linear_model.Lasso()
	evaluate(reg, X, y)



def evaluate_elastic_net(X, y):
	print "Evaluating elastic net regression..."
	reg = sklearn.linear_model.ElasticNet()
	evaluate(reg, X, y)



def evaluate_bayesian_ridge(X, y):
	print "Evaluating Bayesian ridge regression..."
	reg = sklearn.linear_model.BayesianRidge()
	evaluate(reg, X, y)



def evaluate_sgd(X, y):
	print "Evaluating SGD..."
	reg = sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
	evaluate(reg, X, y)



def evaluate_polynomial(X, y):
	print "Evaluating Polynomial regression..."
	X_poly = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(X)
	reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
	evaluate(reg, X_poly, y)



def evaluate_kernel_ridge(X, y):
	print "Evaluating Kernel ridge regression..."
	reg = sklearn.kernel_ridge.KernelRidge()
	evaluate(reg, X, y)



def evaluate_svm_linear(X, y):
	print "Evaluating SVM regression (linear kernel)..."
	reg = sklearn.svm.SVR(kernel="linear")
	evaluate(reg, X, y)



def evaluate_svm_poly(X, y):
	print "Evaluating SVM regression (polynomial kernel)..."
	reg = sklearn.svm.SVR(kernel="poly")
	evaluate(reg, X, y)



def evaluate_svm_rbf(X, y):
	print "Evaluating SVM regression (RBF kernel)..."
	reg = sklearn.svm.SVR(kernel="rbf")
	evaluate(reg, X, y)



def evaluate_mlp(X, y):
	print "Evaluating MLP regression..."
	reg = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(30,), max_iter=500)
	evaluate(reg, X, y)



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "usage: python regress.py [infile] [config]"
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	df = pd.read_csv(sys.argv[1], sep="\t")
	df_cate = df[config["categorical"]]
	X = df[config["numerical"]]
	y = df[config["output"][0]]

	# apply standard scaler
	X = pd.DataFrame(sklearn.preprocessing.scale(X), X.index, X.columns)

	# remove samples with high-variance output
	mask = df[config["output"][1]] < abs(df[config["output"][0]])
	X = X[mask]
	y = y[mask]

	methods = [
		evaluate_linear,
		evaluate_ridge,
		evaluate_lasso,
		evaluate_elastic_net,
		evaluate_bayesian_ridge,
		evaluate_sgd,
		evaluate_polynomial,
		evaluate_kernel_ridge,
		evaluate_svm_linear,
		evaluate_svm_poly,
		evaluate_svm_rbf,
		evaluate_mlp
	]

	for method in methods:
		method(X, y)
