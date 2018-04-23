from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import pandas as pd
import numpy as np
import time
import operator

def majority_vote(neighbors):
	map = {}
	maximum = ( '', 0 ) # (element, occurences)
	
	for n in range(len(neighbors)):
		if n in map: 
			map[n] += 1
		else:
			map[n] = 1
		if map[n] > maximum[1]:
			maximum = (n, map[n])
	return maximum
	
def k_nearest_neighbors(X,y,cv,k):

	#Latent Semantic Indexing
	components = 40
	lsi = TruncatedSVD(n_components = components)
	X = lsi.fit_transform(X)

	kf = KFold(n_splits = 10)
	precision_score = 0
	recall_score = 0 
	f1_score =0
	accuracy_score = 0

	cv = 0
	if (cv == 0):
		N = len(X)
		T = int(N*0.66)
		
		X_train = X[:T]
		y_train = y[:T]
		X_test = X[T:]
		y_test = y[T:]
		
		predictions = []
		for i in range(len(X_test)):
			#Get Nearest Neighbors
			distances =[]
			for j in range(len(X_train)):
				dist = np.sqrt(np.sum(np.square(X_test[i,:] - X_train[j, :])))
				distances.append([dist, j])
			distances = sorted(distances)
			
			neighbors = []
			for j in range(k):
				neighbors.append(y_train[distances[j][1]])
			
			nearest_neighbor = Counter(neighbors).most_common(1)[0][0]

			predictions.append(nearest_neighbor)
		accuracy_score = metrics.accuracy_score(y_test, predictions)
		precision_score =  metrics.precision_score(y_test, predictions, average='macro')
		recall_score =  metrics.recall_score(y_test, predictions, average='macro')
		f1_score =  metrics.f1_score(y_test, predictions, average='macro')
	else:
		print("Performing Cross Validation")

		for train, test in tqdm(kf.split(X)):
			X_train, X_test = X[train], X[test]
		   	y_train, y_test = y[train], y[test]

		predictions = []
		for i in range(len(X_test)):
			#Get Nearest Neighbors
			distances =[]
			for j in range(len(X_train)):
				dist = np.sqrt(np.sum(np.square(X_test[i,:] - X_train[j, :])))
				distances.append([dist, j])
			distances = sorted(distances)
			
			neighbors = []
			for j in range(k):
				neighbors.append(y_train[distances[j][1]])
			
			nearest_neighbor = Counter(neighbors).most_common(1)[0][0]

			predictions.append(nearest_neighbor)
		accuracy_score += metrics.accuracy_score(y_test, predictions)
		precision_score +=  metrics.precision_score(y_test, predictions, average='macro')
		recall_score +=  metrics.recall_score(y_test, predictions, average='macro')
		f1_score +=  metrics.f1_score(y_test, predictions, average='macro')
		
	return accuracy_score, precision_score, recall_score, f1_score






