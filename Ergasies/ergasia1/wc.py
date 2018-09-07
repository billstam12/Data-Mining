import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud



def create_WordCloud(text, stopwords, filename):
	wc = WordCloud(background_color="white",max_words=400,stopwords=stopwords,width=800,height=400)
	wc.generate(text)
	wc.to_file("WordClouds/"+filename+".png")

train_data = pd.read_csv('datasets/train_set.csv',sep="\t")


A = np.array(train_data)

text_p = ""
text_ft = ""
text_f = ""
text_t = ""
text_b = ""

for i in range(0,A.shape[0]):

	if A[i][4] == "Politics":
		text_p += " " + A[i][3]

	elif A[i][4] == "Film":
		text_f += " " + A[i][3]

	elif A[i][4] == "Football":
		text_ft += " " + A[i][3]

	elif A[i][4] == "Technology":
		text_t += " " + A[i][3]

	elif A[i][4] == "Business":
		text_b += " " + A[i][3]

text_file = open("stopwords", "r")
stp = text_file.read().splitlines()
stopwords = ENGLISH_STOP_WORDS.union(stp)

create_WordCloud(text_p, stopwords, "Politics")
create_WordCloud(text_f, stopwords, "Film")
create_WordCloud(text_ft, stopwords, "Football")
create_WordCloud(text_t, stopwords, "Technology")
create_WordCloud(text_b, stopwords, "Business")
