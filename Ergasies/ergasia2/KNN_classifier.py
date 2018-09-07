from sklearn import metrics
from collections import Counter
import sys
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
AVG_EARTH_RADIUS = 6371 #km



	
def haversine(lon1,lat1,lon2,lat2):
	
		
	lon1, lat1, lon2, lat2, = map(radians, (lon1, lat1, lon2, lat2))
	
	#calculate haversine distance
	
	lon = lon2 - lon1
	lat = lat2 - lat1
	
	d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lon * 0.5) ** 2
	h = 2* AVG_EARTH_RADIUS * asin(sqrt(d))
	
	return h

def DTW(search, target):
	n = len(search)
	m = len(target)
	dtw = np.zeros((n,m))
	
	for i in range(1,n):
		dtw[i][0] = float("inf")
	for i in range(1,m):
		dtw[0][i] = float("inf")
	dtw[0][0] = 0
	
	
	for i in range(1, n):
		for j in range(1, m):
			cost = haversine(search[i][1], search[i][2], target[j][1], target[j][2])
			dtw [i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
			
	return dtw[n-1][m-1]

def predict(X_train,y_train,x_test,k):
	distances = []

	for i in range(len(X_train)):
		distances.append([DTW(X_train[i],x_test),i])

	distances = sorted(distances)

	classes = []
	for i in range(k):
		ind = distances[i][1]
		classes.append(y_train[ind])

	return Counter(classes).most_common(1)[0][0]


def k_nearest_neighbors(X_train,y_train,x_test,predictions,k):
	for i in range(len(x_test)):
		predictions.append(predict(X_train,y_train,x_test[i],k))
	
	
