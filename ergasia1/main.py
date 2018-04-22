from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
import os 
from sklearn import metrics
import pandas as pd 
import numpy as np 
import time
from NB_classifier import naive_bayes 
from RF_classifier import random_forest
from SVM_classifier import support_vector_machine
from KNN_classifier import k_nearest_neighbors

def EvaluateMetricCSV(accuracy, precision, recall, f1):

    d = {'Statistic Measure': pd.Series(["Accuracy", "Precision", "Recall", "F-Measure"]),
         'Naive Bayes': pd.Series([accuracy_list[0], precision_list[0], recall_list[0], f1_list[0]]),
         'Random Forest': pd.Series([accuracy_list[1], precision_list[1], recall_list[1], f1_list[1]]),
         'SVM': pd.Series([accuracy_list[2], precision_list[2], recall_list[2], f1_list[2]]),
         'KNN': pd.Series([accuracy_list[3], precision_list[3], recall_list[3], f1_list[3]])}

    df = pd.DataFrame(d)
    df.to_csv('results/EvaluationMetric_10fold.csv', sep='\t', index=False,
            columns=['Statistics', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN'])


train_data = pd.read_csv('datasets/train_set.csv', sep="\t")
text_file = open("stopwords", "r")
stp = text_file.read().splitlines()
vctrzr = TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS.union(stp))
X = vctrzr.fit_transform(train_data['Content'] + 3*train_data['Title'])

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

cv = 0 #Cross Validation Flag
accuracy_list = []
precision_list =[]
recall_list = []
f1_list = []

######################################################################
###########################Naive-Bayes################################
######################################################################

accuracy, precision, recall, f1 = naive_bayes(X,y,cv)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)

######################################################################
###########################Random Forest##############################
######################################################################

accuracy, precision, recall, f1 = random_forest(X,y,cv)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)

######################################################################
########################SVM with Parameter Choosing###################
######################################################################


accuracy, precision, recall, f1 = support_vector_machine(X,y,cv)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)


######################################################################
########################K-Nearest Neighbors###########################
######################################################################

k = 7 #Use multiples of 7 as k
accuracy, precision, recall, f1 = k_nearest_neighbors(X,y,cv,k)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)

EvaluateMetricCSV(accuracy_list, precision_list, recall_list, f1_list)



