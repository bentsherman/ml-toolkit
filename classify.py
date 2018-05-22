import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
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



def confusion_matrix(y_true, y_pred, classes):
	cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=classes)

	sns.heatmap(cnf_matrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
	plt.ylabel("Expected")
	plt.xlabel("Measured")
	plt.show()



def roc_curve(y_true, y_score, classes):
	n_classes = len(classes)

	# compute ROC curve and auc for each class
	fpr = {}
	tpr = {}
	auc = {}

	for i in xrange(n_classes):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true[:, i], y_score[:, i])
		auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

	# aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in xrange(n_classes)]))

	# interpolate all ROC curves at these points
	mean_tpr = np.zeros_like(all_fpr)
	for i in xrange(n_classes):
		mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

	# compute average tpr
	mean_tpr /= n_classes

	colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue"])
	for i, color in zip(xrange(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, label="%s (area = %0.2f)" % (classes[i], auc[i]))

	plt.plot([0, 1], [0, 1], "k--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.title("Receiver operating characteristics")
	plt.legend(loc="lower right")
	plt.show()



def evaluate(model, X, y):
	# compute one-hot labels
	classes = list(set(y))
	classes.sort()

	y_bin = sklearn.preprocessing.label_binarize(y, classes)

	# predict output
	if hasattr(model, "decision_function"):
		score_method = "decision_function"
	else:
		score_method = "predict_proba"

	y_score = sklearn.model_selection.cross_val_predict(model, X, y, cv=5, n_jobs=-1, method=score_method)
	y_pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=5, n_jobs=-1)

	# compute metrics
	acc = sklearn.metrics.accuracy_score(y, y_pred)
	f1 = sklearn.metrics.f1_score(y, y_pred, average="micro")

	print "acc = %8.3f, f1 = %8.3f" % (acc, f1)

	# plot confusion matrix
	confusion_matrix(y, y_pred, classes)

	# plot ROC curve
	roc_curve(y_bin, y_score, classes)



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
