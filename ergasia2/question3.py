import pandas as pd
import numpy as np
from ast import literal_eval
from KNN_classifier import k_nearest_neighbors

trainSet = pd.read_csv('datasets/train_set.csv', converters={"Trajectory": literal_eval})


testSet = pd.read_csv('datasets/test_set_a2.csv', sep=";", converters={"Trajectory": literal_eval})

X = []
y = []
k = 5 #nearest neighbors
for i in range(len(trainSet)):
	x = np.array(trainSet['Trajectory'][i])
	X.append(x)
	y.append(trainSet['journeyPatternId'][i])

test = []

for i in range(len(testSet)):
	x = np.array(testSet['Trajectory'][i])
	test.append(x)

X = np.array(X)
y = np.array(y)


predictions = []
k_nearest_neighbors(X,y,test,predictions,k)

