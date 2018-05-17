import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree
import sys



def evaluate(reg, X, y):
	# predict output
	y_pred = sklearn.model_selection.cross_val_predict(reg, X, y, cv=5)

	# compute metrics
	r, p = scipy.stats.pearsonr(y, y_pred)
	r2 = sklearn.metrics.r2_score(y, y_pred)

	print "r = %8.3f, r^2 = %8.3f" % (r, r2)

	# plot correlation of predicted and expected output
	sns.jointplot(y, y_pred, kind="reg")
	plt.show()



def create_linear(X, y):
	print "Evaluating linear regression..."
	return sklearn.linear_model.LinearRegression()



def create_ridge(X, y):
	print "Evaluating ridge regression..."
	return sklearn.linear_model.Ridge()



def create_lasso(X, y):
	print "Evaluating lasso regression..."
	return sklearn.linear_model.Lasso()



def create_elastic_net(X, y):
	print "Evaluating elastic net regression..."
	return sklearn.linear_model.ElasticNet()



def create_bayesian_ridge(X, y):
	print "Evaluating Bayesian ridge regression..."
	return sklearn.linear_model.BayesianRidge()



def create_sgd(X, y):
	print "Evaluating SGD..."
	return sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3)



def create_polynomial(X, y):
	print "Evaluating Polynomial regression..."
	return sklearn.pipeline.Pipeline([
		("poly", sklearn.preprocessing.PolynomialFeatures(degree=2)),
		("linear", sklearn.linear_model.LinearRegression(fit_intercept=False))
	])



def create_kernel_ridge(X, y):
	print "Evaluating Kernel ridge regression..."
	return sklearn.kernel_ridge.KernelRidge()



def create_svm_linear(X, y):
	print "Evaluating SVM regression (linear kernel)..."
	return sklearn.svm.LinearSVR()



def create_svm_poly(X, y):
	print "Evaluating SVM regression (polynomial kernel)..."
	return sklearn.svm.SVR(kernel="poly")



def create_svm_rbf(X, y):
	print "Evaluating SVM regression (RBF kernel)..."
	return sklearn.svm.SVR(kernel="rbf")



def create_decision_tree(X, y):
	print "Evaluating decision tree regression..."
	return sklearn.tree.DecisionTreeRegressor()



def create_mlp(X, y):
	print "Evaluating MLP regression..."
	return sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(30,), max_iter=500)



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "usage: python regress.py [infile] [config]"
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	# load data, extract X and y
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

	# evaluate each regressor
	methods = [
		create_linear,
		create_ridge,
		create_lasso,
		create_elastic_net,
		create_bayesian_ridge,
		create_sgd,
		create_polynomial,
		create_kernel_ridge,
		create_svm_linear,
		create_svm_poly,
		create_svm_rbf,
		create_decision_tree,
		create_mlp
	]

	for method in methods:
		reg = method(X, y)
		evaluate(reg, X, y)
		print
