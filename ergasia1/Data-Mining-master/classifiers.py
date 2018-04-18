from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import pandas as pd
import numpy as np
import time

from KNN_Classifier import kNearestNeighbor


def EvaluateMetricCSV(accuracy, precision, recall, f1):

    d = {'Statistic Measure': pd.Series(["Accuracy", "Precision", "Recall", "F-Measure"]),
         'Naive Bayes': pd.Series([accuracy[0], precision[0], recall[0], f1[0]]),
         'Random Forest': pd.Series([accuracy[1], precision[1], recall[1], f1[1]]),
         'SVM': pd.Series([accuracy[2], precision[2], recall[2], f1[2]]),
         'KNN': pd.Series([accuracy[4], precision[4], recall[4], f1[4]]),
         'Stochastic Gradient Descent': pd.Series([accuracy[3], precision[3], recall[3], f1[3]])}

    df = pd.DataFrame(d)
    df.to_csv('Produced_Files/EvaluationMetric_10fold.csv', sep='\t', index=False,
            columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'Stochastic Gradient Descent'])


dataset = pd.read_csv('datasets/train_set.csv', sep="\t")
train_data = dataset[0:2000]


option = 1

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

# vectorization of data
my_additional_stop_words = ['people']
# vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(my_additional_stop_words))
X = vectorizer.fit_transform(train_data['Content']).toarray()


start_time = time.time()

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
predictions = []

components = 100
k = 30

# CLASSIFICATION
if option > 0:
    for i in range(0, 4):
        if i == 0:
            print("Multinomial NB")
            # nmf_model = NMF(n_components=components)
            # X = nmf_model.fit_transform(X)
            X1 = X[:1500]
            X2 = X[1500:]

            clf = MultinomialNB()
        elif i == 1:
            print("Random Forest Classifier")
            lsi_model = TruncatedSVD(n_components=components)
            X = lsi_model.fit_transform(X)
            X1 = X[:1500]
            X2 = X[1500:]

            clf = RandomForestClassifier()

        elif i == 2:
            print("SVM SVC")
            # svc = svm.SVC()
            # param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
            # clf = GridSearchCV(svc, param_grid)

            clf = svm.SVC(kernel='linear', C=1, gamma=1)

        elif i == 3:
            print("Stochastic Gradient Descent")
            clf = SGDClassifier()

        clf.fit(X1, y[:1500])
        predictions = clf.predict(X2)

        predicted_categories = le.inverse_transform(predictions)
        precision_score.append(metrics.precision_score(y[1500:], predictions, average='micro'))
        recall_score.append(metrics.recall_score(y[1500:], predictions, average='micro'))
        f1_score.append(metrics.f1_score(y[1500:], predictions, average='micro'))
        accuracy_score.append(metrics.accuracy_score(y[1500:], predictions))

    print("KNN k = ", k)
    predictions = []
    kNearestNeighbor(X1, y[:1500], X2, predictions, k)

    # transform the list into an array
    predictions = np.asarray(predictions)
    predicted_categories = le.inverse_transform(predictions)
    precision_score.append(metrics.precision_score(y[1500:], predictions, average='micro'))
    recall_score.append(metrics.recall_score(y[1500:], predictions, average='micro'))
    f1_score.append(metrics.f1_score(y[1500:], predictions, average='micro'))
    accuracy_score.append(metrics.accuracy_score(y[1500:], predictions))

    EvaluateMetricCSV(accuracy_score, precision_score, recall_score, f1_score)


else:
    if option < 0:

        if option == -1:
            print("Random Forest Classifier")
            lsi_model = TruncatedSVD(n_components=components)
            X = lsi_model.fit_transform(X)
            clf = RandomForestClassifier()

        elif option == -2:
            print("Multinomial NB")
            # nmf_model = NMF(n_components=components)
            # X = nmf_model.fit_transform(X)
            clf = MultinomialNB()

        elif option == -3:
            print("SVM SVC")
            lsi_model = TruncatedSVD(n_components=components)
            X = lsi_model.fit_transform(X)
            clf = svm.SVC(kernel='linear', C=1, gamma=1)
            # svc = svm.SVC()
            # param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
            # clf = GridSearchCV(svc, param_grid)

        elif option == -4:
            lsi_model = TruncatedSVD(n_components=components)
            X = lsi_model.fit_transform(X)
            clf = SGDClassifier()

        X1 = X[:1500]
        X2 = X[1500:]

        clf.fit(X1, y[:1500])
        predictions = clf.predict(X2)

    elif option == 0:

        print("KNN k = ", k)

        lsi_model = TruncatedSVD(n_components=components)
        X = lsi_model.fit_transform(X)
        X1 = X[:1500]
        X2 = X[1500:]

        kNearestNeighbor(X1, y[:1500], X2, predictions, k)

        # transform the list into an array
        predictions = np.asarray(predictions)

    predicted_categories = le.inverse_transform(predictions)
    print(metrics.classification_report(y[1500:], predictions, target_names=list(le.classes_)))
    print("Time %f" % (time.time() - start_time))
