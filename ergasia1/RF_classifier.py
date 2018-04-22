from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import csv
import time



def random_forest(X,y,cv):

	#Latent Semantic Indexing
	components = 100
	lsi = TruncatedSVD(n_components = components)
	X = lsi.fit_transform(X)


	if (cv == 0):
		N = len(X)
		T = int(N*0.66)
		X_train = X[:T]
		y_train = y[:T]
		X_test = X[T:]
		y_test = y[T:]
		
		clf = RandomForestClassifier()
		clf.fit(X_train,y_train)

		predictions = clf.predict(X_test)
		precision_score =  metrics.precision_score(y_test, predictions, average='macro')
		recall_score =  metrics.recall_score(y_test, predictions, average='macro')
		f1_score =  metrics.f1_score(y_test, predictions, average='macro')
		accuracy_score = metrics.accuracy_score(y_test, predictions)
	else:	
		######################################################################
		#Cross Validation
		kf = KFold(n_splits = 10)
		precision_score = 0
		recall_score = 0 
		f1_score =0
		accuracy_score = 0

		for train, test in tqdm(kf.split(X)):
			X_train = np.array(X)[train]
			y_train = np.array(y)[train]
			X_test = np.array(X)[test]
			y_test = np.array(y)[test]
			clf = RandomForestClassifier()
			clf.fit(X_train,y_train)

			predictions = clf.predict(X_test)
			print "Printing stats"
			precision_score +=  metrics.precision_score(y_test, predictions, average='macro')
			recall_score +=  metrics.recall_score(y_test, predictions, average='macro')
			f1_score +=  metrics.f1_score(y_test, predictions, average='macro')
			accuracy_score += metrics.accuracy_score(y_test, predictions)
			
		accuracy_score /= 10
		f1_score /= 10
		recall_score /= 10
		precision_score /= 10
		
	
	return accuracy_score, precision_score, recall_score, f1_score



