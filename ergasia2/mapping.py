import pandas as pd 
from ast import literal_eval
import gmplot

train_set = pd.read_csv('train_set.csv', 
						converters = {"Trajectory": literal_eval},
						index_col='tripId')
jpid = ['00160001', '00161001', '083A1001', '015B1001', '077A0001'] #5 patternIds
traj = train_set['Trajectory'].loc[train_set['journeyPatternId'].isin(jpid)].head(5)
print traj



"""
gmap = gmplot.GoogleMapPlotter("Dublin")

# Polygon
golden_gate_park_lats, golden_gate_park_lons = zip(*[
    (37.771269, -122.511015),
    (37.773495, -122.464830),
    (37.774797, -122.454538),
    (37.771988, -122.454018),
    (37.773646, -122.440979),
    (37.772742, -122.440797),
    (37.771096, -122.453889),
    (37.768669, -122.453518),
    (37.766227, -122.460213),
    (37.764028, -122.510347),
    (37.771269, -122.511015)
    ])
gmap.plot(golden_gate_park_lats, golden_gate_park_lons, 'cornflowerblue', edge_width=10)

gmap.draw("map.html")
"""