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



def evaluate(model, X, y):
	# predict output
	y_pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=5)

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



def create_logistic_regression():
	return sklearn.linear_model.LogisticRegression()



def create_sgd_hinge():
	return sklearn.linear_model.SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3)



def create_sgd_log():
	return sklearn.linear_model.SGDClassifier(loss="log", max_iter=1000, tol=1e-3)



def create_sgd_perceptron():
	return sklearn.linear_model.SGDClassifier(loss="perceptron", max_iter=1000, tol=1e-3)



def create_lda():
	return sklearn.discriminant_analysis.LinearDiscriminantAnalysis()



def create_qda():
	return sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()



def create_svm_linear():
	return sklearn.svm.LinearSVC()



def create_svm_poly():
	return sklearn.svm.SVC(kernel="poly")



def create_svm_rbf():
	return sklearn.svm.SVC(kernel="rbf")



def create_svm_sigmoid():
	return sklearn.svm.SVC(kernel="sigmoid")



def create_knn():
	return sklearn.neighbors.KNeighborsClassifier()



def create_naive_bayes():
	return sklearn.naive_bayes.GaussianNB()



def create_decision_tree():
	return sklearn.tree.DecisionTreeClassifier()



def create_mlp():
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
		("logistic regression", create_logistic_regression),
		("SGD (hinge loss)", create_sgd_hinge),
		("SGD (log loss)", create_sgd_log),
		("SGD (perceptron loss)", create_sgd_perceptron),
		("LDA classifier", create_lda),
		("QDA classifier", create_qda),
		("SVM classifier (linear kernel)", create_svm_linear),
		("SVM classifier (poly kernel)", create_svm_poly),
		("SVM classifier (RBF kernel)", create_svm_rbf),
		("SVM classifier (sigmoid kernel)", create_svm_sigmoid),
		("k-NN classifier", create_knn),
		("naive Bayes classifier", create_naive_bayes),
		("decision tree classifier", create_decision_tree),
		("MLP classifier", create_mlp)
	]

	for (name, create_model) in methods:
		print "Evaluating %s..." % (name)
		model = create_model()
		evaluate(model, X, y)
		print
