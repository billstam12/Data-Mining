import pandas as pd 
from ast import literal_eval
from collections import Counter
import dtw
import numpy as np 
import gmplot
from math import radians, cos, sin, asin, sqrt

AVG_EARTH_RADIUS = 6371 #km

train_set = pd.read_csv('datasets/train_set.csv', 
						converters = {"Trajectory": literal_eval},
						index_col='tripId')

test_set = pd.read_csv('datasets/test_set_a1.csv', sep =';', converters = {"Trajectory": literal_eval})
						
def haversine(lon1,lat1,lon2,lat2):
	
		
	lon1, lat1, lon2, lat2, = map(radians, (lon1, lat1, lon2, lat2))
	
	#calculate haversine distance
	
	lon = lon2 - lon1
	lat = lat2 - lat1
	
	d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lon * 0.5) ** 2
	h = 2* AVG_EARTH_RADIUS * asin(sqrt(d))
	
	return h
	
def lcs(search , target):
    
	m = len(search)
	n = len(target)
	
	
	L = [[None]*(n+1) for i in xrange(m+1)]
	
	for i in range(m+1):
		for j in range(n+1):
			x = haversine(search[i-1][1], search[i-1][2], target[j-1][1], target[j-1][2]) 
            
			if i == 0 or j == 0 :
				L[i][j] = 0
			elif x <= 0.2:
				L[i][j] = L[i-1][j-1]+1
			else:
				L[i][j] = max(L[i-1][j] , L[i][j-1])
    
	return L[m][n]

train_traj = train_set['Trajectory']

test_traj = test_set['Trajectory']
# Here we iterate through each instance of the table and
# for each of the elements in every row we remove the time element
# thus we keep only the coordinates.


for i in test_traj:
	lc = []
	num = 0
	for j in train_traj[:100]:
		lc.append([lcs(i,j),num])
		num +=1
	lc = sorted(lc, reverse=True)
	
	for k in range(0,5):
		print lc[k];
