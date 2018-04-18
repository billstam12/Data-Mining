import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

train_data = pd.read_csv('datasets/train_set.csv', sep = "\t")
train_data = train_data[0:100]

A = np.array(train_data)
categories = set(train_data['Category'])
no_of_c = len(categories);