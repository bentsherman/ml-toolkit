import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
import sklearn.discriminant_analysis
import sklearn.ensemble
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



def confusion_matrix(y_true, y_score, y_pred, classes):
	cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=classes)

	sns.heatmap(cnf_matrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
	plt.ylabel("Expected")
	plt.xlabel("Measured")
	plt.title("Confusion Matrix")
	plt.show()



def roc_curve(y_true, y_score, y_pred, classes):
	n_classes = len(classes)

	# determine whether labels are multi-class
	if n_classes == 2:
		# condense scores to single-class scores
		y_score = y_score[:, 1]

		# compute FPR, TPR, and auc
		fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
		auc = sklearn.metrics.auc(fpr, tpr)

		# plot ROC curve
		plt.plot(fpr, tpr, label="area = %0.2f" % (auc))
	else:
		# compute one-hot labels
		y_bin = sklearn.preprocessing.label_binarize(y_true, classes)

		# compute FPR, TPR, and auc for each class
		fpr = {}
		tpr = {}
		auc = {}

		for i in range(n_classes):
			fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_bin[:, i], y_score[:, i])
			auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

		# plot ROC curve for each class
		for i in range(n_classes):
			plt.plot(fpr[i], tpr[i], label="%s (area = %0.2f)" % (classes[i], auc[i]))

	plt.plot([0, 1], [0, 1], "k--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.title("Receiver operating characteristics")
	plt.legend(loc="lower right")
	plt.show()



def precision_recall_curve(y_true, y_score, y_pred, classes):
	n_classes = len(classes)

	# plot iso-f1 curves
	f1_scores = np.linspace(0.2, 0.8, num=4)

	for f1_score in f1_scores:
		x = np.linspace(0.01, 1)
		y = f1_score * x / (2 * x - f1_score)
		plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
		plt.annotate("f1=%0.1f" % (f1_score), xy=(0.9, y[45] + 0.02))

	# determine whether labels are multi-class
	if n_classes == 2:
		# condense scores to single-class scores
		y_score = y_score[:, 1]

		# compute precision, recall, and average precision
		precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
		average_precision = sklearn.metrics.average_precision_score(y_true, y_score)

		# plot precision-recall curve
		plt.plot(recall, precision, label="area = %0.2f" % (average_precision))
	else:
		# compute one-hot labels
		y_bin = sklearn.preprocessing.label_binarize(y_true, classes)

		# compute precision, recall, and average precision for each class
		precision = {}
		recall = {}
		average_precision = {}

		for i in range(n_classes):
			precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_bin[:, i], y_score[:, i])
			average_precision[i] = sklearn.metrics.average_precision_score(y_bin[:, i], y_score[:, i])

		# plot precision-recall curve for each class
		for i in range(n_classes):
			plt.plot(recall[i], precision[i], label="%s (area = %0.2f)" % (classes[i], average_precision[i]))

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall")
	plt.legend(loc="lower left")
	plt.show()



def evaluate(model, X, y):
	# compute class names
	classes = list(set(y))
	classes.sort()

	# compute class scores for X using cross-validation
	if hasattr(model, "decision_function"):
		score_method = "decision_function"
	else:
		score_method = "predict_proba"

	y_score = sklearn.model_selection.cross_val_predict(model, X, y, cv=5, n_jobs=-1, method=score_method)

	# compute predicted labels from class scores
	y_pred = [classes[y_i.argmax()] for y_i in y_score]

	# compute scores
	scores = [
		("acc", sklearn.metrics.accuracy_score(y, y_pred)),
		("f1", sklearn.metrics.f1_score(y, y_pred, average="weighted"))
	]

	print("  scores:")

	for (name, value) in scores:
		print("    %-4s = %8.3f" % (name, value))

	# create plots
	plots = [
		("confusion matrix", confusion_matrix),
		("roc curve", roc_curve),
		("precision-recall curve", precision_recall_curve)
	]

	print("  plots:")

	for (name, create_plot) in plots:
		print("    %s" % (name))
		create_plot(y, y_score, y_pred, classes)



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



def create_bagging():
	return sklearn.ensemble.BaggingClassifier()



def create_random_forest():
	return sklearn.ensemble.RandomForestClassifier()



def create_ada_boost():
	return sklearn.ensemble.AdaBoostClassifier()



def create_gradient_boosting():
	return sklearn.ensemble.GradientBoostingClassifier()



def create_mlp():
	return sklearn.neural_network.MLPClassifier(max_iter=500)



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: python regress.py [infile] [config]")
		sys.exit(1)

	config = json.load(open(sys.argv[2]))

	# load data, extract X and y
	df = pd.read_table(sys.argv[1])
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
		("Bagging classifier", create_bagging),
		("random forest classifier", create_random_forest),
		("AdaBoost classifier", create_ada_boost),
		("gradient tree boosting classifier", create_gradient_boosting),
		("MLP classifier", create_mlp)
	]

	for (name, create_model) in methods:
		print("Evaluating %s..." % (name))
		model = create_model()
		evaluate(model, X, y)
		print
