from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import time


components = 40

train_data = pd.read_csv('datasets/train_set.csv', sep="\t")
test_data = pd.read_csv('datasets/test_set.csv', sep ="\t")
cross_val = 1
text_file = open("stopwords", "r")
stp = text_file.read().splitlines()
vctrzr = TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS.union(stp))
X = vctrzr.fit_transform(train_data['Content'] + train_data['Title'])

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

lsi = TruncatedSVD(n_components = components)
X = lsi.fit_transform(X)
N = len(X)
T = int(N*0.66)

X_test = vctrzr.fit_transform(test_data['Content'] + test_data['Title'])
X_test = lsi.fit_transform(X_test)
	
scaler =  StandardScaler()
scaler.fit(X)
scaler.fit(X_test)

if (cross_val == 0):
	
	X_train = X[:T]
	X_test = X[T:]
	y_train = y[:T]
	y_test = y[T:]
	clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
	clf.fit(X_train,y_train)
	
	predictions = clf.predict(X_test)
	#predictions = le.inverse_transform(predictions)
	
	accuracy_score = metrics.accuracy_score(y_test, predictions)
	precision_score =  metrics.precision_score(y_test, predictions, average='macro')
	recall_score =  metrics.recall_score(y_test, predictions, average='macro')
	f1_score =  metrics.f1_score(y_test, predictions, average='macro')
	print accuracy_score, precision_score, recall_score, f1_score
else:	
	print("Performing Cross Validation")
	
	clf = MLPClassifier(hidden_layer_sizes=(25,25,25), warm_start=True, max_iter=500, alpha=0.0001, activation="tanh",solver='sgd', verbose=10,learning_rate_init=0.01,learning_rate="adaptive", random_state=21,tol=0.000000001)
	clf.fit(X,y)
	#scores = cross_val_score(clf, X, y, cv = 10)
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	
	
	predictions = clf.predict(X_test)

	headers = ['Id', 'Category']
	results = pd.DataFrame(zip(test_data.Id, le.inverse_transform(predictions)), columns=headers)
	results.to_csv('results/testSet_categories.csv', index=False)
	








