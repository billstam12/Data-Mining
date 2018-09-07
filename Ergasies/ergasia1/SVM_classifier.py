from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import csv
import time
from sklearn import svm



def support_vector_machine(X,y,cv):
	#Latent Semantic Indexing
	components = 40
	lsi = TruncatedSVD(n_components = components)
	X = lsi.fit_transform(X)


	#Split Set
	if(cv==0):
		N = len(X)
		T = int(N*0.66)
		X_train = X[:T]
		y_train = y[:T]
		X_test = X[T:]
		y_test = y[T:]

	#Here we used GridSearchCV to find the best parameters for SVM
		#svc = svm.SVC()
		#param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000, 10000], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
		#clf = GridSearchCV(svc, param_grid)
		clf = svm.SVC(kernel='rbf', C=10000, gamma=0.01)
		clf.fit(X_train,y_train)
		#print clf.best_params_
		#{'kernel': 'rbf', 'C': 10000, 'gamma': 0.01}

		predictions = clf.predict(X_test)
		precision_score =  metrics.precision_score(y_test, predictions, average='macro')
		recall_score =  metrics.recall_score(y_test, predictions, average='macro')
		f1_score =  metrics.f1_score(y_test, predictions, average='macro')
		accuracy_score = metrics.accuracy_score(y_test, predictions)
	else:
		#Cross Validation
		kf = KFold(n_splits = 10)
		precision_score = 0
		recall_score = 0 
		f1_score =0
		accuracy_score = 0

		for train, test in tqdm(kf.split(X)):
			X_train, X_test = X[train], X[test]
		   	y_train, y_test = y[train], y[test]
			
			clf = svm.SVC(kernel='rbf', C=10000, gamma=0.01)
			clf.fit(X_train,y_train)

			predictions = clf.predict(X_test)
			precision_score +=  metrics.precision_score(y_test, predictions, average='macro')
			recall_score +=  metrics.recall_score(y_test, predictions, average='macro')
			f1_score +=  metrics.f1_score(y_test, predictions, average='macro')
			accuracy_score += metrics.accuracy_score(y_test, predictions)
			
		accuracy_score /= 10
		f1_score /= 10
		recall_score /= 10
		precision_score /= 10

	return accuracy_score, precision_score, recall_score, f1_score


