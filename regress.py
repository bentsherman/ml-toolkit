import json
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sys



def evaluate_cv(reg, X, y):
	scores = sklearn.model_selection.cross_val_score(reg, X, y, cv=5)
	print "r = %8.3f, r^2 = %8.3f +/- %.3f" % (math.sqrt(max(0, scores.mean())), scores.mean(), scores.std())



def evaluate_plot(reg, X, y):
	for i in xrange(5):
		# split dataset into train / test sets
		X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2)

		# fit regression model
		print reg.fit(X_train, y_train)

		# predict output
		y_pred = reg.predict(X_test)

		# plot correlation of predicted and expected output
		sns.jointplot(y_test, y_pred, kind="reg")
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
	return sklearn.svm.SVR(kernel="linear")



def create_svm_poly(X, y):
	print "Evaluating SVM regression (polynomial kernel)..."
	return sklearn.svm.SVR(kernel="poly")



def create_svm_rbf(X, y):
	print "Evaluating SVM regression (RBF kernel)..."
	return sklearn.svm.SVR(kernel="rbf")



def create_mlp(X, y):
	print "Evaluating MLP regression..."
	return sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(30,), max_iter=500)



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
		create_mlp
	]

	for method in methods:
		reg = method(X, y)
		evaluate_cv(reg, X, y)
		print

	for method in methods:
		reg = method(X, y)
		evaluate_plot(reg, X, y)
		print
