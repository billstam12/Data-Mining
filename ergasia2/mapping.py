import pandas as pd 
from ast import literal_eval
import gmplot

train_set = pd.read_csv('datasets/train_set.csv', 
						converters = {"Trajectory": literal_eval},
						index_col='tripId')
jpid = ['00160001', '00161001', '083A1001', '015B1001', '077A0001'] #5 patternIds
traj = train_set['Trajectory'].loc[train_set['journeyPatternId'].isin(jpid)].head(5)

traj_list = traj.values


lats_1 = []
lats_2 = []
lats_3 = []
lats_4 = []
lats_5 = []

lons_1 = []
lons_2 = []
lons_3 = []
lons_4 = []
lons_5 = []

# Polygon
for i in range(0,5):
	for j in traj_list[i]:
		if(i == 0):
			lats_1.append(j[2])
			lons_1.append(j[1])
		elif(i == 1):
			lats_2.append(j[2])
			lons_2.append(j[1])
		elif(i == 2):
			lats_3.append(j[2])
			lons_3.append(j[1])
		elif(i == 3):
			lats_4.append(j[2])
			lons_4.append(j[1])
		elif(i == 4):
			lats_5.append(j[2])
			lons_5.append(j[1])

			
gmap1 = gmplot.GoogleMapPlotter(53.383015, -6.237581, 10)			
gmap1.plot(lats_1,lons_1,'cornflowerblue',edge_width=10)
gmap1.draw("maps/map1.html")

gmap2 = gmplot.GoogleMapPlotter(53.383015, -6.237581, 10)			
gmap2.plot(lats_2,lons_2,'cornflowerblue',edge_width=10)
gmap2.draw("maps/map2.html")

gmap3 = gmplot.GoogleMapPlotter(53.383015, -6.237581, 10)			
gmap3.plot(lats_3,lons_3,'cornflowerblue',edge_width=10)
gmap3.draw("maps/map3.html")

gmap4 = gmplot.GoogleMapPlotter(53.383015, -6.237581, 10)		
gmap4.plot(lats_4,lons_4,'cornflowerblue',edge_width=10)
gmap4.draw("maps/map4.html")

gmap5 =  gmplot.GoogleMapPlotter(53.383015, -6.237581, 10)		
gmap5.plot(lats_5,lons_5,'cornflowerblue',edge_width=10)
gmap5.draw("maps/map5.html")

