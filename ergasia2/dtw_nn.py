import pandas as pd 
from ast import literal_eval
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

def DTW(search, target):
	n = len(search)
	m = len(target)
	dtw = numpy.zeros((n,m))
	
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


traj = train_set['Trajectory']
train_traj = traj.values

test_traj = test_set['Trajectory']
# Here we iterate through each instance of the table and
# for each of the elements in every row we remove the time element
# thus we keep only the coordinates.



for i in range(0,len(test_set)):
	coords = []
	for j in test_traj[i]:
		j.pop(0)
		coords.append(j)
	#print coords
	
	coords2 = []
#print train_traj









