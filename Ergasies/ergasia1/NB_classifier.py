from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import csv
import time


def naive_bayes(X,y,cv):

	if(cv==0):
		N = 2000
		T = int(N*0.66)
		X_train = X[:T]
		y_train = y[:T]
		X_test = X[T:]
		y_test = y[T:]
		
		clf = MultinomialNB()
		clf.fit(X_train,y_train)

		predictions = clf.predict(X_test)
		
		precision_score =  metrics.precision_score(y_test, predictions, average='macro')
		recall_score =  metrics.recall_score(y_test, predictions, average='macro')
		f1_score =  metrics.f1_score(y_test, predictions, average='macro')
		accuracy_score = metrics.accuracy_score(y_test, predictions)
	else:
		kf = KFold(n_splits = 10)
		precision_score = 0
		recall_score = 0 
		f1_score =0
		accuracy_score = 0
		for train, test in tqdm(kf.split(X)):
			X_train, X_test = X[train], X[test]
		   	y_train, y_test = y[train], y[test]
			
			clf = MultinomialNB()
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



