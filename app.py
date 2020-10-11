import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
data_1 = pd.read_csv('/content/drive/My Drive/imdb dataset/IMDB Dataset.csv')
def clean1(text):
  soup = BeautifulSoup(text,'html.parser')
  a = soup.get_text()
  return a
data_1['review'] = data_1['review'].apply(clean1)
def clean2(text):
  text = re.sub('\[[^]]*\]',' ',text)
  text = re.sub('[^a-zA-Z]',' ',text)
  return text
data_1['review'] = data_1['review'].apply(clean2)
def clean3(text):
  text = text.lower()
  return text
data_1['review'] = data_1['review'].apply(clean3)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def clean4(text):
  a = []
  text = text.split()
  for i in text:
    if i not in stopwords.words('english'):
      a.append(i)
  return a
data_1['review'] = data_1['review'].apply(clean4)
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lem = WordNetLemmatizer()
def clean5(text):
  text = ' '.join(text)
  return text
def clean6(text):
  text = [lem.lemmatize(word) for word in text]
  text = clean5(text)
  return text
data_1['review'] = data_1['review'].apply(clean6)
X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(data_1['review'],data_1['sentiment'],test_size = 0.2)
le_1 = LabelEncoder()
y_train_1 = le_1.fit_transform(y_train_1)
y_test_1 = le_1.transform(y_test_1)
corpus_train_1 = []
corpus_test_1 = []
for i in X_train_1.index:
  temp = X_train_1[i]
  corpus_train_1.append(temp)
for j in X_test_1.index:
  temp1 = X_test_1[j]
  corpus_test_1.append(temp1)
X_train_1 = corpus_train_1
X_test_1 = corpus_test_1

st.title("IMDB movie review")
st.subheader('Count Vectorizer')
st.write('This project is based on Linear SVC classifier')
text_model = Pipeline([('cv2',CountVectorizer()),('model',LinearSVC(C = 0.5,max_iter = 10000))])
text_model.fit(X_train_1,y_train_1)
message = st.text_area("Enter Text","the movie was good,the actors did a wonderful job, but plot seemed repetitive")
op = text_model.predict([message])
if st.button("Predict"):
  st.title(op)