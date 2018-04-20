from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import numpy as np
import pandas as pd
import time

components = 100
accuracy_score = 0

train_data = pd.read_csv('datasets/train_set.csv', sep="\t")
N = len(train_data)
T = int(N*0.66)
cv = 0

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = vectorizer.fit_transform(train_data['Title'],train_data['Content']).toarray()
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
lsi_model = TruncatedSVD(n_components=components)
X = lsi_model.fit_transform(X)

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])

y = le.transform(train_data["Category"])
if (cv == 0):
	X_train = X[:T]
	X_test = X[T:]
	y_train = y[:T]
	y_test = y[T:]
	
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	clf = MLPClassifier(activation = 'relu', solver='sgd', warm_start=True, alpha=1e-5, max_iter = 1000, hidden_layer_sizes=(N/2,N/4,N/8),learning_rate ="adaptive", learning_rate_init = 0.01, momentum = 0.01, random_state=1)
	clf.fit(X_train,y_train)
	predictions = clf.predict(X_test)
	accuracy_score = metrics.accuracy_score(y_test, predictions)

else:
	print("Performing Cross Validation")
	scaler =  StandardScaler()
	clf = MLPClassifier(activation = 'relu', solver='sgd', warm_start=True, alpha=1e-5, max_iter = 1000, hidden_layer_sizes=(6, N/2),learning_rate ="adaptive", learning_rate_init = 0.01, momentum = 0.01, random_state=1)
	pipeline = make_pipeline(scaler, clf)
	scores = cross_val_score(pipeline, X, y)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	










