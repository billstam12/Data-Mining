from collections import Counter
import numpy as np
import pandas as pd
from dtw import dtw
from math import radians, cos, sin, asin, sqrt
import time
from scipy.spatial import cKDTree
AVG_EARTH_RADIUS = 6371 #km
from sklearn.neighbors import KNeighborsClassifier

def haversine(lon1,lat1,lon2,lat2):
	
		
	lon1, lat1, lon2, lat2, = map(radians, (lon1, lat1, lon2, lat2))
	
	#calculate haversine distance
	
	lon = lon2 - lon1
	lat = lat2 - lat1
	
	d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lon * 0.5) ** 2
	h = 2* AVG_EARTH_RADIUS * asin(sqrt(d))
	
	return h


def predict(X_train, y_train, X_test, k):
	distances = []
	neighbors = []
	
	for i in range(len(X_train)):
		distance = dtw(X_train[i], X_test, dist = lambda search, target: haversine
			(search[1], search[2], target[1], target[2]))[0]
	
	
		distances.append([distance, i])
	
	# sort the list
	distances = sorted(distances)
	
	# make a list of the k neighbors
	for i in range(k):
		index = distances[i][1]
		neighbors.append(y_train[index])
	
	#Perform majority vote with counter
	return Counter(neighbors).most_common(1)[0][0]

			

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i], k))
