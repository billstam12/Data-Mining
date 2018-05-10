import pandas as pd 
from ast import literal_eval
import dtw
import gmplot
from math import radians, cos, sin, asin, sqrt

AVG_EARTH_RADIUS = 6371 #km

train_set = pd.read_csv('datasets/train_set.csv', 
						converters = {"Trajectory": literal_eval},
						index_col='tripId')
test_set = pd.read_csv('datasets/test_set_a1.csv', 
						converters = {"Trajectory": literal_eval},
						index_col='tripId')
						
def haversine(point1, point2):
	
	lon1, lat1 = point1
	lon2, lat2 = point2
	
	lon1, lat1, lon2, lat2, = map(radians, (lon1, lat1, lon2, lat2))
	
	#calculate haversine distance
	
	lon = lon2 - lon1
	lat = lat2 - lat1
	
	d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lon * 0.5) ** 2
	h = 2* AVG_EARTH_RADIUS * asin(sqrt(d))
	
	return h 
	
traj = train_set['Trajectory']

traj_list = traj.values

distances = []

