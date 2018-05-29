import pandas as pd
from ast import literal_eval
import numpy as np

from KNN import kNearestNeighbor
train_set = pd.read_csv('datasets/train_set.csv', converters={"Trajectory": literal_eval})
train_set = train_set[:100]
test_set = pd.read_csv('datasets/test_set_a2.csv', sep="\t", converters={"Trajectory": literal_eval})['Trajectory']
tripIDs = list(range(1, 6))

X = []
y = []
test = []
for i in range(len(train_set)):
	x = np.array(train_set["Trajectory"][i])
	X.append(x)
	y.append(train_set['journeyPatternId'][i])

for i in range(len(test_set)):
	x = np.array(test_set[i])
	test.append(x)


X = np.array(X)
y = np.array(y)
k = 5
predictions = []
kNearestNeighbor(X, y, test, predictions, k)
print(predictions)
