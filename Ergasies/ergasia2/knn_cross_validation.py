import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from KNN import kNearestNeighbor
from ast import literal_eval
import time

train_set = pd.read_csv('datasets/train_set.csv', converters={"Trajectory": literal_eval})
N = len(train_set)
T = int(N*0.1)
train_set = train_set[:T] #get 5% of train set


X = []
y = []

for i in range(len(train_set)):
	x = np.array(train_set["Trajectory"][i])
	X.append(x)
	y.append(train_set['journeyPatternId'][i])

	
X = np.array(X)
y = np.array(y)
splits = StratifiedKFold(n_splits = 10)
splits.get_n_splits(X,y)
accuracy = 0
k = 5
start = time.time()
for train, test in splits.split(X,y):
	X_train, X_test = X[train], X[test]
	y_train, y_test = y[train], y[test]
	
	predictions = []
	
	kNearestNeighbor(X_train, y_train, X_test, predictions, k)
	
	accuracy += metrics.accuracy_score(y_test, predictions)

elapsed_time = time.time() - start
print("Accuracy = ", accuracy/10)
print("Time = ", elapsed_time)