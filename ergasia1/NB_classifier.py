from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import csv
import time



train_data = pd.read_csv('datasets/train_set.csv', sep="\t")

vectorizer = CountVectorizer(stop_words = ENGLISH_STOP_WORDS)
X = vectorizer.fit_transform(train_data['Title'],train_data['Content']).toarray()

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])


#NO VECTORIZATION FOR NAIVE BAYES, IT CANT HANDLE NEGATIVE NUMBERS

cv = 0
######################################################################
###########################Naive-Bayes################################
######################################################################
if(cv==0):
	N = len(X)
	T = int(N*0.66)
	X_train = X[:T]
	y_train = y[:T]
	X_test = X[T:]
	y_test = y[T:]
	
	clf = MultinomialNB()
	clf.fit(X_train,y_train)

	predictions = clf.predict(X_test)
	
	precision_score =  metrics.precision_score(y_test, predictions, average='macro')
	recall_score =  metrics.recall_score(y_test, predictions, average='macro')
	f1_score =  metrics.f1_score(y_test, predictions, average='macro')
	accuracy_score = metrics.accuracy_score(y_test, predictions)
else:
	kf = KFold(n_splits = 10)
	precision_score = 0
	recall_score = 0 
	f1_score =0
	accuracy_score = 0
	for train, test in tqdm(kf.split(X)):
		X_train = np.array(X)[train]
		y_train = np.array(y)[train]
		X_test = np.array(X)[test]
		y_test = np.array(y)[test]
		
		clf = MultinomialNB()
		clf.fit(X_train,y_train)

		predictions = clf.predict(X_test)
		print "Printing stats"
		precision_score +=  metrics.precision_score(y_test, predictions, average='macro')
		recall_score +=  metrics.recall_score(y_test, predictions, average='macro')
		f1_score +=  metrics.f1_score(y_test, predictions, average='macro')
		accuracy_score += metrics.accuracy_score(y_test, predictions)
		

	accuracy_score /= 10
	f1_score /= 10
	recall_score /= 10
	precision_score /= 10
		
print "Printing Naive Bayes statistics"
print precision_score, recall_score, f1_score, accuracy_score

d = {'Statistics': pd.Series(["Accuracy", "Precision", "Recall", "F-Measure"]),
	'Naive Bayes':pd.Series([accuracy_score,precision_score,recall_score,f1_score])}

df = pd.DataFrame(d)
df.to_csv('results/EvaluationMetric_10fold.csv', sep='\t', index=False,
    columns=['Statistics', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'Stochastic Gradient Descent'])

