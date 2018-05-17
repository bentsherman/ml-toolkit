import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree
import sys



def evaluate(clf, X, y):
	# predict output
	y_pred = sklearn.model_selection.cross_val_predict(clf, X, y, cv=5)

	# compute metrics
	acc = sklearn.metrics.accuracy_score(y, y_pred)
	f1 = sklearn.metrics.f1_score(y, y_pred, average="micro")

	print "acc = %8.3f, f1 = %8.3f" % (acc, f1)

	# plot confusion matrix
	classes = list(set(y))
	cnf_matrix = sklearn.metrics.confusion_matrix(y, y_pred, labels=classes)

	sns.heatmap(cnf_matrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
	plt.ylabel("y")
	plt.xlabel("y_pred")
	plt.show()



def create_logistic_regression(X, y):
	print "Evaluating logistic regression..."
	return sklearn.linear_model.LogisticRegression()



def create_sgd_hinge(X, y):
	print "Evaluating SGD (hinge loss)..."
	return sklearn.linear_model.SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3)



def create_sgd_log(X, y):
	print "Evaluating SGD (log loss)..."
	return sklearn.linear_model.SGDClassifier(loss="log", max_iter=1000, tol=1e-3)



def create_sgd_perceptron(X, y):
	print "Evaluating SGD (perceptron loss)..."
	return sklearn.linear_model.SGDClassifier(loss="perceptron", max_iter=1000, tol=1e-3)



def create_lda(X, y):
	print "Evaluating LDA classifier..."
	return sklearn.discriminant_analysis.LinearDiscriminantAnalysis()



def create_qda(X, y):
	print "Evaluating QDA classifier..."
	return sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()



def create_svm_linear(X, y):
	print "Evaluating SVM classifier (linear kernel)..."
	return sklearn.svm.LinearSVC()



def create_svm_poly(X, y):
	print "Evaluating SVM classifier (polynomial kernel)..."
	return sklearn.svm.SVC(kernel="poly")



def create_svm_rbf(X, y):
	print "Evaluating SVM classifier (RBF kernel)..."
	return sklearn.svm.SVC(kernel="rbf")



def create_svm_sigmoid(X, y):
	print "Evaluating SVM classifier (sigmoid kernel)..."
	return sklearn.svm.SVC(kernel="sigmoid")



def create_knn(X, y):
	print "Evaluating k-NN classifier..."
	return sklearn.neighbors.KNeighborsClassifier()



def create_naive_bayes(X, y):
	print "Evaluating naive Bayes classifier..."
	return sklearn.naive_bayes.GaussianNB()



def create_decision_tree(X, y):
	print "Evaluating decision tree classifier..."
	return sklearn.tree.DecisionTreeClassifier()



def create_mlp(X, y):
	print "Evaluating MLP classifier..."
	return sklearn.neural_network.MLPClassifier()



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

	# evaluate each classifier
	methods = [
		create_logistic_regression,
		create_sgd_hinge,
		create_sgd_log,
		create_sgd_perceptron,
		create_lda,
		create_qda,
		create_svm_linear,
		create_svm_poly,
		create_svm_rbf,
		create_svm_sigmoid,
		create_knn,
		create_naive_bayes,
		create_decision_tree,
		create_mlp
	]

	for method in methods:
		clf = method(X, y)
		evaluate(clf, X, y)
		print
