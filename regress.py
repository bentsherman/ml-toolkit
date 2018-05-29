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



def evaluate(model, X, y):
	# predict output
	y_pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=5, n_jobs=-1)

	# compute metrics
	r, p = scipy.stats.pearsonr(y, y_pred)
	r2 = sklearn.metrics.r2_score(y, y_pred)

	print "r = %8.3f, r^2 = %8.3f" % (r, r2)

	# plot correlation of expected and predicted output
	limits = (min(min(y), min(y_pred)), max(max(y), max(y_pred)))

	sns.jointplot(y, y_pred, kind="reg", xlim=limits, ylim=limits)
	plt.xlabel("Expected")
	plt.ylabel("Measured")
	plt.show()



def create_linear():
	return sklearn.linear_model.LinearRegression()



def create_ridge():
	return sklearn.linear_model.Ridge()



def create_lasso():
	return sklearn.linear_model.Lasso()



def create_elastic_net():
	return sklearn.linear_model.ElasticNet()



def create_bayesian_ridge():
	return sklearn.linear_model.BayesianRidge()



def create_sgd():
	return sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3)



def create_polynomial():
	return sklearn.pipeline.Pipeline([
		("poly", sklearn.preprocessing.PolynomialFeatures(degree=2)),
		("linear", sklearn.linear_model.LinearRegression(fit_intercept=False))
	])



def create_kernel_ridge():
	return sklearn.kernel_ridge.KernelRidge()



def create_svm_linear():
	return sklearn.svm.LinearSVR()



def create_svm_poly():
	return sklearn.svm.SVR(kernel="poly")



def create_svm_rbf():
	return sklearn.svm.SVR(kernel="rbf")



def create_decision_tree():
	return sklearn.tree.DecisionTreeRegressor()



def create_mlp():
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
	output_sd = "%s_SD" % config["output"][0]
	if output_sd in df.columns:
		mask = df[output_sd] < 1.5
		X = X[mask]
		y = y[mask]

	# evaluate each regressor
	methods = [
		("linear regression", create_linear),
		("ridge regression", create_ridge),
		("lasso regression", create_lasso),
		("elastic net regression", create_elastic_net),
		("Bayesian ridge regression", create_bayesian_ridge),
		("SGD regression", create_sgd),
		("Polynomial regression", create_polynomial),
		("Kernel ridge regression", create_kernel_ridge),
		("SVM regression (linear kernel)", create_svm_linear),
		("SVM regression (polynomial kernel)", create_svm_poly),
		("SVM regression (RBF kernel)", create_svm_rbf),
		("decision tree regression", create_decision_tree),
		("MLP regression", create_mlp)
	]

	for (name, create_model) in methods:
		print "Evaluating %s..." % (name)
		model = create_model()
		evaluate(model, X, y)
		print
