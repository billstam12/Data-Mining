from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd


components = 40

train_data = pd.read_csv('datasets/train_set.csv', sep="\t")
test_data = pd.read_csv('datasets/test_set.csv', sep ="\t")

# V E C T O R I Z E
text_file = open("stopwords", "r")
stp = text_file.read().splitlines()
vctrzr = TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS.union(stp))
X = vctrzr.fit_transform(train_data['Content'] + 10*train_data['Title'])

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

# L S A 
lsi = TruncatedSVD(n_components = components)
X = lsi.fit_transform(X)

X_test = vctrzr.transform(test_data['Content'] + 10*test_data['Title'])
X_test = lsi.transform(X_test)

# S T A N D A R D I Z E	
scaler =  StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print("Performing Cross Validation")

clf = MLPClassifier(hidden_layer_sizes=(45,45,45),  max_iter=1000, alpha=0.0001, activation="relu", solver='sgd', verbose=10,learning_rate_init=0.01,learning_rate="adaptive", random_state=21,tol=0.000000001)
clf.fit(X,y)
#scores = cross_val_score(clf, X, y, cv = 10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	

predictions = clf.predict(X_test)

headers = ['Id', 'Category']
results = pd.DataFrame(zip(test_data.Id, le.inverse_transform(predictions)), columns=headers)
results.to_csv('results/testSet_categories.csv', index=False)