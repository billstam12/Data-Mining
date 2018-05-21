import pandas as pd 
from ast import literal_eval
from collections import Counter
import dtw
import numpy as np 
import gmplot
from math import radians, cos, sin, asin, sqrt
import time 

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
	

train_traj = train_set['Trajectory']

test_traj = test_set['Trajectory']
# Here we iterate through each instance of the table and
# for each of the elements in every row we remove the time element
# thus we keep only the coordinates.

index = 1
for traj in test_traj:
	distances = []
	num = 0
	traj = np.array(traj)
	lats = []
	lons = []
	
	#Get coordinates
	for item in traj:
		lons.append(item[1])
		lats.append(item[2])
	
	# PLOT THE TEST ROUTE
	gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)
	gmap.plot(lats, lons, 'cornflowerblue', edge_width=5)
	gmap.draw("results2/Test_Route_"+str(index)+".html")
	
	#Calculate dtw and create distances list
	start_time = time.time()
	for j in train_traj[:1000]:
		distances.append([DTW(traj,j),num])
		num +=1	#This counts the position of the neighbor
	distances = sorted(distances)
	
	elapsed_time = time.time() - start_time
	
	for k in range(0,5):
		#print distances[k];
		lats = []
		lons = []
		
		#Calculate coordinates of nearest neighbors and plot them too
		for l in train_traj[distances[k][1]]:
			lons.append(l[1])
			lats.append(l[2])
			
		gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)
		gmap.plot(lats, lons, 'cornflowerblue', edge_width=5)
		gmap.draw("results2/Neighbor_"+str(index)+str(distances[k][1])+".html")

	fl = open('results2/final/final_'+str(index)+'.html','w')
	message = """
<!DOCTYPE html>
<html>
	<body>
	<style>
		.center {
			display: block;
			margin-left: auto;
			margin-right: auto;
			width: 50%;
		}
	</style>
	<table>
		<tr>
			<td><iframe src = "../Test_Route_"""+str(index)+""".html"></iframe></td>
			<td><iframe src = "../Neighbor_"""+str(index)+str(distances[0][1])+""".html"></iframe></td>
			<td><iframe src = "../Neighbor_"""+str(index)+str(distances[1][1])+""".html"></iframe></td>
		</tr>
		<tr>
			<td>Test Trip """+str(index)+"""</td>
			<td>Neighbor 1</td>
			<td>Neighbor 2</td>
		</tr>
		<tr>
			<td>Dt= """+str(elapsed_time)+"""sec</td>
			<td>JP_ID: """+train_set['journeyPatternId'][distances[0][1]]+"""</td>
			<td>JP_ID: """+train_set['journeyPatternId'][distances[1][1]]+"""</td>
		</tr>
		<tr>
			<td></td>
			<td>DTW:  """+str(distances[0][0])+"""</td>
			<td>DTW:  """+str(distances[1][0])+"""</td>
		</tr>
		<tr>
			<td><iframe src = "../Neighbor_"""+str(index)+str(distances[2][1])+""".html"></iframe></td>
			<td><iframe src = "../Neighbor_"""+str(index)+str(distances[3][1])+""".html"></iframe></td>
			<td><iframe src = "../Neighbor_"""+str(index)+str(distances[4][1])+""".html"></iframe></td>
		</tr>
		<tr>
			<td>Neighbor 3</td>
			<td>Neighbor 4</td>
			<td>Neighbor 5</td>
		</tr>
		<tr>
			<td>JP_ID: """+train_set['journeyPatternId'][distances[2][1]]+"""</td>
			<td>JP_ID: """+train_set['journeyPatternId'][distances[3][1]]+"""</td>
			<td>JP_ID: """+train_set['journeyPatternId'][distances[4][1]]+"""</td>
		</tr>
		<tr>
			<td>DTW:  """+str(distances[2][0])+"""</td>
			<td>DTW:  """+str(distances[3][0])+"""</td>
			<td>DTW:  """+str(distances[4][0])+"""</td>
		</tr>
	</table>
</body>
</html>"""
	fl.write(message)
	fl.close()
	index+=1
	
			


